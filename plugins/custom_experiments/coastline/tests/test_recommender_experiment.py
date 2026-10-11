# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Behavioural tests for the COASTLINE recommender custom experiments.

Every test here drives the COASTLINE engine, which ships as this plugin's optional
``coastline`` extra. The whole module is skipped when the extra is absent -- as it
is in ado's CI, which does not install the autogluon/torch stack the engine pulls,
and where ``test_experiment_registration.py`` covers the engine-free contract
instead.

The tests call each experiment's ``_original_func`` so they exercise the
recommendation logic directly, without Ray, the actuator or the measurement
pipeline. They default to ``feasibility="rules"``, the plugin's AutoConf-free
backend -- the engine's structural guards (a positive GPU count, a per-device batch
of at least 1) plus its empirical per-device token budget -- which keeps them off the
classifier: no AutoConf model is loaded and no prediction is made. It does not keep
them off the AutoGluon/torch stack, which any environment running this module already
has: ``ado-autoconf`` is a core dependency of ``coastline-recommender``, not an extra.
Two tests exercise the AutoConf path itself and skip when AutoConf will not load.
"""

import logging
import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ado_coastline import recommender_experiment
from ado_coastline._bridge import import_facade
from ado_coastline.recommender_experiment import (
    coastline_min_gpu_recommender,
    coastline_recommender,
)

ENGINE_MISSING_REASON = (
    "the COASTLINE engine is not installed; install the extra with "
    'uv pip install -e "plugins/custom_experiments/coastline[coastline]"'
)

# Skip the whole module unless every symbol the plugin's bridge imports is available.
for _engine_module in (
    "coastline",
    "coastline.sdk.models.context",
    "coastline.sdk.predictors.feasibility.autoconf",
):
    pytest.importorskip(_engine_module, reason=ENGINE_MISSING_REASON)

#: The engine's empirical per-device token ceiling [tokens], read from the engine
#: rather than restated here: these tests pin the plugin's contract ("a load over the
#: budget is refused"), which must keep holding if the budget is ever refitted.
EMPIRICAL_OOM_TOKEN_BUDGET: int = pytest.importorskip(
    "coastline.sdk.constants", reason=ENGINE_MISSING_REASON
).EMPIRICAL_OOM_TOKEN_BUDGET

#: The plugin's README, located from this file so the test is independent of the
#: working directory: ``<plugin>/tests/`` -> ``<plugin>/README.md``.
README_PATH = Path(__file__).resolve().parents[1] / "README.md"

#: A Kavier-calibrated performance model used across the tests.
SUPPORTED_MODEL = "granite-3.3-8b"

#: The GPU the engine-free-feasibility tests run on.
SUPPORTED_GPU = "NVIDIA-A100-SXM4-80GB"

#: The GPU the AutoConf path runs on, as in the README's worked example.
AUTOCONF_GPU = "NVIDIA-A100-80GB-PCIe"

#: Inputs shared by the comparison tests, so only the policy differs between runs.
COMMON_INPUTS = {
    "model_name": SUPPORTED_MODEL,
    "method": "full",
    "gpu_model": SUPPORTED_GPU,
    "tokens_per_sample": 2048,
    "batch_size": 16,
    "max_gpus": 8,
    "feasibility": "rules",
}

#: The same job with a per-device load above the empirical token budget (32 x 2048 =
#: 65,536 tokens/device). Only the batch size moves, so the budget is the one thing
#: that can explain a different verdict. Every test using it asserts that first.
OVER_BUDGET_INPUTS = {**COMMON_INPUTS, "batch_size": 32}

#: The AutoConf path's inputs: the README's worked example, a LoRA job on PCIe A100s.
AUTOCONF_INPUTS = {
    "model_name": SUPPORTED_MODEL,
    "method": "lora",
    "gpu_model": AUTOCONF_GPU,
    "tokens_per_sample": 2048,
    "batch_size": 16,
    "max_gpus": 8,
    "feasibility": "autoconf",
}

#: Rated board power [W] of one GPU, per the vendor spec sheets -- and per the
#: ``max_power_w`` of the Kavier catalog the engine's power model is calibrated
#: against. Used only as a plausibility envelope, never as an expected value.
GPU_BOARD_POWER_WATTS = {
    SUPPORTED_GPU: 400.0,
    AUTOCONF_GPU: 300.0,
}

#: The fraction of its boards' rated power a busy fine-tuning job must draw. Wide
#: enough to survive a recalibration of the power model -- these recommendations land
#: near 0.54 -- and tight enough to catch a per-GPU figure published as the job's,
#: which for a 4-GPU job comes in at 0.13 and for an 8-GPU job at 0.07.
PLAUSIBLE_BOARD_POWER_FRACTION = (0.25, 1.05)

#: The presets whose pick is wide enough for the job-level power guard to bite. The
#: energy and min_gpu policies choose a single GPU, where the per-GPU and job-level
#: figures coincide and the unit bug is invisible.
MULTI_GPU_PRESETS = ["balanced", "performance"]


def unwrap(experiment: Callable[..., Any]) -> Callable[..., Any]:
    """Return the undecorated experiment function, bypassing ado's validation.

    Args:
        experiment: A ``@custom_experiment``-decorated function.

    Returns:
        The wrapped original function, so it can be called without Ray or the
        actuator.
    """
    return getattr(experiment, "_original_func", experiment)


def per_device_tokens(inputs: dict[str, Any]) -> int:
    """Return the tokens one device holds for a set of experiment inputs.

    Args:
        inputs: Experiment keyword arguments, carrying ``batch_size`` (per device) and
            ``tokens_per_sample``.

    Returns:
        ``batch_size x tokens_per_sample``, the quantity the empirical budget bounds.
    """
    return int(inputs["batch_size"]) * int(inputs["tokens_per_sample"])


def facade_recorder(
    *, autoconf_loads: bool = True, recommends: bool = True
) -> tuple[Callable[[], tuple[type, type, type]], list[dict[str, Any]]]:
    """Build a stand-in ``import_facade`` that records how the engine is constructed.

    The plugin's feasibility contract is a property of the engine *construction*: which
    keyword arguments it passes per backend. Recording them is what lets the AutoConf
    path be asserted on a machine without AutoConf, where the real probe would degrade
    the run to ``rules`` and hide the very question being asked -- and, with
    ``autoconf_loads=False``, what lets the degraded path be asserted on a machine that
    does have AutoConf. The recorded engine returns one canned, self-consistent
    single-GPU recommendation; its numbers are never asserted on.

    Args:
        autoconf_loads: What the stubbed AutoConf probe reports, i.e. whether the
            plugin should see AutoConf as loadable.
        recommends: Whether the recorded engine finds anything. ``False`` makes it
            return an empty list, which is how the plugin's "no feasible candidate"
            path is reached without depending on any real verdict.

    Returns:
        ``(import_facade_stub, constructions)``, where ``constructions`` gains one dict
        of constructor keyword arguments per engine the plugin builds.
    """
    constructions: list[dict[str, Any]] = []
    _, SystemContext, _ = import_facade()

    class RecordingCoastline:
        """Records its construction, then answers with one canned recommendation.

        The constructor mirrors ``coastline.Coastline``'s signature, defaults and all,
        so it is a faithful stand-in: a plugin that stopped passing
        ``empirical_oom_guard`` would record the engine's own default here, exactly as
        the real engine would have applied it.
        """

        def __init__(
            self,
            predictor: str = "kavier",
            *,
            energy: str = "kavier_power",
            feasibility: str = "autoconf",
            empirical_oom_guard: bool = False,
        ) -> None:
            constructions.append(
                {
                    "predictor": predictor,
                    "energy": energy,
                    "feasibility": feasibility,
                    "empirical_oom_guard": empirical_oom_guard,
                }
            )

        def recommend(
            self, workload: dict[str, object], **kwargs: object
        ) -> list[SimpleNamespace]:
            """Return a single 1-GPU recommendation, shaped as the plugin reads it.

            Args:
                workload: The workload the plugin built; ignored.
                **kwargs: The grid and policy arguments; ignored.

            Returns:
                One canned recommendation, enough for the plugin to publish a row, or
                nothing at all when this recorder was built with ``recommends=False``.
            """
            if not recommends:
                return []
            return [
                SimpleNamespace(
                    total_gpus=1,
                    gpus_per_node=1,
                    number_of_nodes=1,
                    predicted_throughput=1.0,
                    metadata={"predicted_power_watts": 1.0, "batch_size": 1},
                )
            ]

    class StubAutoconfChecker:
        """Stands in for COASTLINE's AutoConf availability probe."""

        @staticmethod
        def available() -> bool:
            """Report the availability this recorder was built with."""
            return autoconf_loads

    def import_facade_stub() -> tuple[type, type, type]:
        """Hand back the recording facade in place of the engine's."""
        return RecordingCoastline, SystemContext, StubAutoconfChecker

    return import_facade_stub, constructions


def autoconf_available() -> bool:
    """Report whether COASTLINE can load the AutoConf OOM classifier.

    Returns:
        True when the ``ado-autoconf`` stack is importable and its model loads.
    """
    _, _, autoconf_checker = import_facade()
    return bool(autoconf_checker.available())


def engine_per_gpu_watts(
    recommended: dict[str, Any],
    *,
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    max_gpus: int,
    feasibility: str,
    gpus_per_node: int = 8,
) -> float:
    """Ask the engine directly what one GPU of the recommended layout draws.

    Pinning the grid to the layout's own GPU budget and batch size leaves exactly one
    candidate, so no policy is involved and the recommendation that comes back is that
    layout -- priced by the same predictors the plugin used. Its raw
    ``metadata["predicted_power_watts"]`` is COASTLINE's PER-GPU figure, which is the
    input the plugin has to scale before publishing a job-level number.

    Args:
        recommended: The plugin's published recommendation, whose layout is priced.
        model_name: The LLM, as handed to the experiment.
        method: The fine-tuning method, as handed to the experiment.
        gpu_model: The GPU model, as handed to the experiment.
        tokens_per_sample: Sequence length, as handed to the experiment.
        batch_size: PER-DEVICE batch size, as handed to the experiment.
        max_gpus: The experiment's GPU budget, bounding the context exactly as the
            plugin bounds it.
        feasibility: The feasibility backend the experiment used.
        gpus_per_node: GPUs per node, defaulting to the experiment's own default.

    Returns:
        The engine's per-GPU wattage for the recommended layout.
    """
    Coastline, SystemContext, _ = import_facade()
    context = SystemContext.for_gpus(
        [gpu_model],
        max_gpus=max_gpus,
        gpus_per_node=gpus_per_node,
        max_nodes=max(1, math.ceil(max_gpus / gpus_per_node)),
    )
    engine = Coastline(
        predictor="kavier", energy="kavier_power", feasibility=feasibility
    )
    priced = engine.recommend(
        {
            "llm_model": model_name,
            "fine_tuning_method": method,
            "gpu_model": gpu_model,
            "tokens_per_sample": tokens_per_sample,
            "batch_size": batch_size,
        },
        context=context,
        strategy="multi_objective",
        preset="balanced",
        total_gpus=[recommended["total_gpus"]],
        batch_sizes=[batch_size],
        top_k=1,
        max_gpus=max_gpus,
    )
    assert len(priced) == 1, (
        f"pinning the grid to one budget and one batch size must leave exactly one "
        f"candidate, got {len(priced)}"
    )
    layout = (priced[0].total_gpus, priced[0].gpus_per_node, priced[0].number_of_nodes)
    assert layout == (
        recommended["total_gpus"],
        recommended["gpus"],
        recommended["workers"],
    ), "the probe must price the very layout the plugin recommended"
    watts = (priced[0].metadata or {})["predicted_power_watts"]
    assert watts > 0, "the engine reported no per-GPU power for the recommended layout"
    return float(watts)


def job_power_bounds(gpu_model: str, total_gpus: int) -> tuple[float, float]:
    """Return the watt range a whole job's reported draw has to fall in.

    Args:
        gpu_model: The GPU the job runs on.
        total_gpus: How many of them the job occupies.

    Returns:
        ``(low, high)`` watts, as a fraction of the layout's rated board power.
    """
    rated = GPU_BOARD_POWER_WATTS[gpu_model] * total_gpus
    low, high = PLAUSIBLE_BOARD_POWER_FRACTION
    return low * rated, high * rated


class TestEngineBridge:
    """The bridge hands back the three symbols the experiments drive."""

    def test_import_facade_returns_the_facade_symbols(self) -> None:
        """``import_facade()`` yields Coastline, SystemContext and the OOM checker."""
        from ado_coastline._bridge import import_facade

        facade = import_facade()
        assert len(facade) == 3
        assert [symbol.__name__ for symbol in facade] == [
            "Coastline",
            "SystemContext",
            "AutoconfFeasibilityChecker",
        ]


class TestMultiObjectiveRecommender:
    """``coastline_recommender`` over the AutoConf-free ``rules`` feasible set."""

    def test_balanced_recommendation(self) -> None:
        """The balanced preset returns a usable, self-consistent GPU layout."""
        result = unwrap(coastline_recommender)(
            **COMMON_INPUTS,
            preset="balanced",
        )
        assert result["can_recommend"] is True
        assert result["gpus"] >= 1
        assert result["workers"] >= 1
        assert result["total_gpus"] == result["gpus"] * result["workers"]
        assert result["predicted_throughput"] > 0
        assert result["predicted_power_watts"] > 0
        assert result["feasibility_backend"] == "rules"
        assert result["strategy"] == "coastline_balanced"

    def test_output_contains_every_declared_identifier(self) -> None:
        """A successful recommendation reports every declared output property."""
        declared = {
            target.identifier
            for target in coastline_recommender._experiment.targetProperties
        }
        result = unwrap(coastline_recommender)(
            model_name=SUPPORTED_MODEL,
            method="lora",
            gpu_model="NVIDIA-A100-SXM4-80GB",
            tokens_per_sample=1024,
            batch_size=8,
            feasibility="rules",
        )
        assert result["can_recommend"] is True
        assert declared.issubset(result)

    @pytest.mark.parametrize("preset", ["balanced", "energy", "performance"])
    def test_preset_variants_recommend(self, preset: str) -> None:
        """Each multi-objective preset produces a recommendation and labels itself.

        Args:
            preset: The multi-objective preset under test.
        """
        result = unwrap(coastline_recommender)(**COMMON_INPUTS, preset=preset)
        assert result["can_recommend"] is True
        assert result["strategy"] == f"coastline_{preset}"

    def test_energy_preset_not_more_gpus_than_performance(self) -> None:
        """The energy preset never picks more GPUs than the performance preset.

        Energy weights power (favouring fewer GPUs); performance weights speed. For
        a fixed batch the energy pick's total GPUs must be <= the performance pick's.
        """
        energy = unwrap(coastline_recommender)(**COMMON_INPUTS, preset="energy")
        performance = unwrap(coastline_recommender)(
            **COMMON_INPUTS, preset="performance"
        )
        assert energy["can_recommend"] is True
        assert performance["can_recommend"] is True
        assert energy["total_gpus"] <= performance["total_gpus"]

    def test_unsupported_model_cannot_recommend(self) -> None:
        """An uncalibrated model yields no feasible candidate, not an exception."""
        result = unwrap(coastline_recommender)(
            **{**COMMON_INPUTS, "model_name": "totally-unknown-model-xyz"},
            preset="balanced",
        )
        assert result["can_recommend"] is False


class TestMinimumGPURecommender:
    """``coastline_min_gpu_recommender``, the autoconf min_gpu analogue."""

    def test_min_gpu_recommendation(self) -> None:
        """The min-GPU policy returns the smallest feasible layout with predictions."""
        result = unwrap(coastline_min_gpu_recommender)(**COMMON_INPUTS)
        assert result["can_recommend"] is True
        assert result["total_gpus"] >= 1
        assert result["strategy"] == "coastline_min_gpu"
        assert result["predicted_throughput"] > 0

    def test_min_gpu_not_more_than_balanced(self) -> None:
        """min_gpu must use no more GPUs than the balanced multi-objective pick."""
        min_gpu = unwrap(coastline_min_gpu_recommender)(**COMMON_INPUTS)
        balanced = unwrap(coastline_recommender)(**COMMON_INPUTS, preset="balanced")
        assert min_gpu["can_recommend"] is True
        assert balanced["can_recommend"] is True
        assert min_gpu["total_gpus"] <= balanced["total_gpus"]


class TestAutoconfFeasibilityBackend:
    """The "uses AutoConf" claim, exercised when ado-autoconf is installed."""

    @pytest.mark.skipif(
        not autoconf_available(), reason="ado-autoconf is not installed"
    )
    def test_autoconf_gates_the_grid(self) -> None:
        """With AutoConf installed, the default backend gates the candidate grid."""
        result = unwrap(coastline_recommender)(**AUTOCONF_INPUTS, preset="balanced")
        assert result["can_recommend"] is True
        assert result["feasibility_backend"] == "autoconf"
        assert result["total_gpus"] >= 1


class TestEmpiricalTokenBudgetContract:
    """``rules`` here is the engine's guards *plus* the empirical token budget.

    The plugin promises three distinguishable gates, and this class pins each by the
    verdict it produces rather than by how it is wired:

    - ``autoconf``: the structural guards and then the classifier, with no empirical
      guard on top, so that path stays identical to the runs COASTLINE's published
      results came from.
    - ``rules`` (asked for, or degraded to when AutoConf will not load): structural
      guards plus the per-device token budget, so a run without AutoConf still refuses
      the per-device loads that were observed to OOM. Stricter, therefore, than the
      engine's own bare ``feasibility="rules"``, which has no memory model at all.
    - ``none``: nothing.

    Both experiments share ``_run_recommendation``. The multi-objective experiment holds
    the per-device batch fixed, so the budget vetoes the whole job; ``min_gpu`` splits
    the job's total batch, so the budget picks the first GPU count that brings the
    per-device load under it, and refuses the job only when no count does.
    """

    def test_over_budget_load_is_refused_on_rules(self) -> None:
        """A per-device load above the budget yields no recommendation on ``rules``."""
        assert per_device_tokens(OVER_BUDGET_INPUTS) > EMPIRICAL_OOM_TOKEN_BUDGET, (
            f"this case is only a guard test while its per-device load "
            f"({per_device_tokens(OVER_BUDGET_INPUTS)} tokens) is above the empirical "
            f"budget of {EMPIRICAL_OOM_TOKEN_BUDGET}"
        )

        result = unwrap(coastline_recommender)(**OVER_BUDGET_INPUTS)

        assert result["can_recommend"] is False, (
            f"a load of {per_device_tokens(OVER_BUDGET_INPUTS)} tokens/device is over "
            f"the empirical budget of {EMPIRICAL_OOM_TOKEN_BUDGET} and must be refused; "
            f"the engine's bare 'rules' backend would have accepted it"
        )
        assert result["feasibility_backend"] == "rules", (
            "a refused row must still name the backend that refused it, or a sweep "
            "cannot tell this veto from an AutoConf one"
        )

    def test_min_gpu_splits_an_over_budget_batch_until_it_fits(self) -> None:
        """``min_gpu`` splits the job's total batch over the first GPU count that fits.

        The batch is the job's TOTAL, so on one GPU the load is over the budget, and the
        first power of two that divides the batch and brings the per-device load under
        the budget is the pick, with the batch split evenly.
        """
        total = OVER_BUDGET_INPUTS["batch_size"]
        tokens = OVER_BUDGET_INPUTS["tokens_per_sample"]
        assert total * tokens > EMPIRICAL_OOM_TOKEN_BUDGET
        expected = next(
            g
            for g in (1, 2, 4, 8)
            if total % g == 0 and (total // g) * tokens <= EMPIRICAL_OOM_TOKEN_BUDGET
        )

        result = unwrap(coastline_min_gpu_recommender)(**OVER_BUDGET_INPUTS)

        assert result["can_recommend"] is True
        assert result["feasibility_backend"] == "rules"
        assert result["total_gpus"] == expected
        assert result["recommended_batch_size"] * result["total_gpus"] == total
        assert result["recommended_batch_size"] * tokens <= EMPIRICAL_OOM_TOKEN_BUDGET

    def test_min_gpu_refuses_when_no_split_fits(self) -> None:
        """A total batch of 1 cannot be split, so an over-budget load is refused."""
        inputs = {
            **COMMON_INPUTS,
            "batch_size": 1,
            "tokens_per_sample": EMPIRICAL_OOM_TOKEN_BUDGET + 1,
        }

        result = unwrap(coastline_min_gpu_recommender)(**inputs)

        assert result["can_recommend"] is False
        assert result["feasibility_backend"] == "rules"

    def test_under_budget_load_still_recommends_on_rules(self) -> None:
        """The guard vetoes only over-budget loads: a smaller one still recommends."""
        assert per_device_tokens(COMMON_INPUTS) < EMPIRICAL_OOM_TOKEN_BUDGET, (
            f"this case only shows the guard is selective while its per-device load "
            f"({per_device_tokens(COMMON_INPUTS)} tokens) is below the empirical "
            f"budget of {EMPIRICAL_OOM_TOKEN_BUDGET}"
        )

        result = unwrap(coastline_recommender)(**COMMON_INPUTS, preset="balanced")

        assert result["can_recommend"] is True
        assert result["feasibility_backend"] == "rules"
        assert result["total_gpus"] >= 1

    def test_none_accepts_the_load_rules_refuses(self) -> None:
        """``none`` is truly none: the guard does not leak onto the unchecked backend.

        The same inputs that ``rules`` refuses must come back with a recommendation,
        which is only possible if the empirical guard was not applied.
        """
        assert per_device_tokens(OVER_BUDGET_INPUTS) > EMPIRICAL_OOM_TOKEN_BUDGET

        refused = unwrap(coastline_recommender)(**OVER_BUDGET_INPUTS, preset="balanced")
        unchecked = unwrap(coastline_recommender)(
            **{**OVER_BUDGET_INPUTS, "feasibility": "none"}, preset="balanced"
        )

        assert refused["can_recommend"] is False
        assert unchecked["can_recommend"] is True, (
            f"feasibility='none' must apply no check at all, but a load of "
            f"{per_device_tokens(OVER_BUDGET_INPUTS)} tokens/device -- over the "
            f"empirical budget of {EMPIRICAL_OOM_TOKEN_BUDGET} -- was refused, so the "
            f"guard is leaking onto a backend that is meant to have none"
        )
        assert unchecked["feasibility_backend"] == "none"

    @pytest.mark.parametrize(
        ("requested", "guard_expected"),
        [("autoconf", False), ("rules", True), ("none", False)],
    )
    def test_engine_is_built_with_the_guard_only_for_rules(
        self,
        monkeypatch: pytest.MonkeyPatch,
        requested: str,
        guard_expected: bool,
    ) -> None:
        """Only ``rules`` switches the engine's empirical guard on.

        Asserted on the construction rather than on a verdict because that is the only
        way to see the AutoConf path's setting without installing AutoConf -- and the
        AutoConf path carrying no guard is exactly what keeps it comparable with
        COASTLINE's published runs.

        Args:
            monkeypatch: Used to swap the module's ``import_facade`` for the recorder.
            requested: The feasibility backend asked of the experiment.
            guard_expected: Whether that backend must enable the empirical guard.
        """
        import_facade_stub, constructions = facade_recorder()
        monkeypatch.setattr(recommender_experiment, "import_facade", import_facade_stub)

        result = unwrap(coastline_recommender)(
            **{**COMMON_INPUTS, "feasibility": requested}, preset="balanced"
        )

        assert result["feasibility_backend"] == requested
        assert len(constructions) == 1, (
            f"expected exactly one engine construction, got {len(constructions)}"
        )
        assert constructions[0]["feasibility"] == requested
        assert constructions[0]["empirical_oom_guard"] is guard_expected, (
            f"feasibility={requested!r} must build the engine with "
            f"empirical_oom_guard={guard_expected}, got "
            f"{constructions[0]['empirical_oom_guard']!r}"
        )

    def test_degrading_from_autoconf_still_enables_the_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that falls back to ``rules`` keeps the empirical budget.

        This is the case the contract exists for: without AutoConf the plugin used to
        have no memory model at all, and the degraded run now still refuses the
        per-device loads that were observed to OOM.

        Args:
            monkeypatch: Used to swap the module's ``import_facade`` for a recorder
                whose AutoConf probe reports the classifier as unloadable.
        """
        import_facade_stub, constructions = facade_recorder(autoconf_loads=False)
        monkeypatch.setattr(recommender_experiment, "import_facade", import_facade_stub)

        result = unwrap(coastline_recommender)(
            **{**COMMON_INPUTS, "feasibility": "autoconf"}, preset="balanced"
        )

        assert result["feasibility_backend"] == "rules", (
            "an unloadable AutoConf must degrade the run, and say so on the row"
        )
        assert constructions[0]["feasibility"] == "rules"
        assert constructions[0]["empirical_oom_guard"] is True, (
            "the degraded path is the one that most needs the empirical budget: "
            "without it the run would have no memory model at all"
        )

    def test_degrading_warns_once_even_when_the_row_fails(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A degraded measurement that recommends nothing still warns exactly once.

        A failing row resolves the backend twice -- once for the attempt, once to
        label the ``can_recommend=False`` payload with the backend that was in play --
        and only the attempt may speak. Logging the degrade per resolution would make
        the warning count a function of how many rows failed rather than of how many
        ran degraded, which is the number an operator reads it for.

        Args:
            monkeypatch: Used to swap the module's ``import_facade`` for a recorder
                whose AutoConf probe reports the classifier as unloadable and whose
                engine finds nothing.
            caplog: Captures the plugin logger's records.
        """
        import_facade_stub, _ = facade_recorder(autoconf_loads=False, recommends=False)
        monkeypatch.setattr(recommender_experiment, "import_facade", import_facade_stub)

        with caplog.at_level(logging.WARNING, logger=recommender_experiment.__name__):
            result = unwrap(coastline_recommender)(
                **{**COMMON_INPUTS, "feasibility": "autoconf"}, preset="balanced"
            )

        assert result == {"can_recommend": False, "feasibility_backend": "rules"}, (
            "a degraded row that found nothing must still name the backend it ran on"
        )
        degradations = [
            record.getMessage()
            for record in caplog.records
            if "degrading to the 'rules' backend" in record.getMessage()
        ]
        assert len(degradations) == 1, (
            f"one degraded measurement must log one degradation warning, got "
            f"{len(degradations)}: {degradations}"
        )


class TestDocumentedBudgetMatchesTheEngine:
    """Every place the plugin restates the budget must quote the engine's constant.

    The budget is a *fitted* number, not a design constant: it was selected on the
    OOMs of COASTLINE's calibration campaigns, and a later campaign can refit it. The
    plugin nevertheless spells it out as a literal in five places an operator reads --
    the module docstring, the shipped ``feasibility`` property description (what ``ado
    get experiments`` prints), the :func:`_resolve_feasibility_backend` docstring, the
    degradation warning and the README. That is deliberate for the property
    description, which ado reads at import time, and harmless elsewhere; what is not
    harmless is the plugin quietly going on quoting a figure the engine stopped
    enforcing.

    So these are the binding. A refit of ``EMPIRICAL_OOM_TOKEN_BUDGET`` fails here,
    once per surface that still carries the old number, instead of silently turning
    the plugin's documentation into a lie about what gated a run.
    """

    @pytest.mark.parametrize(
        ("surface", "text"),
        [
            ("the module docstring", recommender_experiment.__doc__),
            (
                "the shipped 'feasibility' property description",
                recommender_experiment.Feasibility.metadata["description"],
            ),
            (
                "the _resolve_feasibility_backend docstring",
                recommender_experiment._resolve_feasibility_backend.__doc__,
            ),
            ("the plugin README", README_PATH.read_text(encoding="utf-8")),
        ],
        ids=[
            "module_docstring",
            "property_description",
            "function_docstring",
            "readme",
        ],
    )
    def test_prose_quotes_the_engine_budget(self, surface: str, text: str) -> None:
        """A documented budget must be the engine's, formatted as the prose writes it.

        Args:
            surface: The document under test, named so a failure says which one to fix.
            text: That document's text.
        """
        quoted = f"{EMPIRICAL_OOM_TOKEN_BUDGET:,}"

        assert quoted in text, (
            f"{surface} must quote the engine's empirical per-device token budget as "
            f"'{quoted}'. The engine's EMPIRICAL_OOM_TOKEN_BUDGET is now "
            f"{EMPIRICAL_OOM_TOKEN_BUDGET}, so that prose describes a gate the plugin "
            f"no longer applies"
        )

    def test_degradation_warning_quotes_the_engine_budget(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning announcing a degrade names the budget that then gates the run.

        This surface matters most of the five: the operator reading it is being told,
        at measurement time, which per-device loads this row was allowed to carry. A
        stale figure there misdescribes a run that already happened.

        Args:
            monkeypatch: Used to swap the module's ``import_facade`` for a recorder
                whose AutoConf probe reports the classifier as unloadable, which is
                what puts the plugin on the degraded path.
            caplog: Captures the plugin logger's records.
        """
        import_facade_stub, _ = facade_recorder(autoconf_loads=False)
        monkeypatch.setattr(recommender_experiment, "import_facade", import_facade_stub)

        with caplog.at_level(logging.WARNING, logger=recommender_experiment.__name__):
            unwrap(coastline_recommender)(
                **{**COMMON_INPUTS, "feasibility": "autoconf"}, preset="balanced"
            )

        degradations = [
            record.getMessage()
            for record in caplog.records
            if "degrading to the 'rules' backend" in record.getMessage()
        ]
        assert len(degradations) == 1, (
            f"the degraded run must log exactly one degradation warning to read the "
            f"budget out of, got {len(degradations)}: {degradations}"
        )
        quoted = f"{EMPIRICAL_OOM_TOKEN_BUDGET:,}"
        assert quoted in degradations[0], (
            f"the degradation warning must quote the engine's empirical per-device "
            f"token budget as '{quoted}', since that budget is what gates the degraded "
            f"run; it says: {degradations[0]}"
        )


class TestJobLevelPowerUnits:
    """Regression guard: the published power is the JOB's draw, not one GPU's.

    COASTLINE reports ``predicted_power_watts`` per GPU -- its power model is a
    single-GPU fit, and it returns the same wattage for every layout of one workload
    -- while ``predicted_throughput`` is already the cluster total. The plugin
    publishes job-level quantities, so it has to scale the power by ``total_gpus``
    and derive ``tokens_per_watt`` from its own two published figures rather than
    from the engine's metadata, which divides a cluster-total throughput by a
    per-GPU wattage.

    Reporting the per-GPU figure unscaled shipped once: it understated an N-GPU
    job's draw by exactly N and inflated ``tokens_per_watt`` by the same factor,
    which is the difference between a recommendation that looks efficient and one
    that is. COASTLINE's own ranking was never affected -- its ``_power_cost``
    already multiplies by ``total_gpus`` -- so nothing but these reported numbers
    would have given the bug away.
    """

    @pytest.mark.parametrize("preset", MULTI_GPU_PRESETS)
    def test_power_is_the_per_gpu_figure_times_total_gpus(self, preset: str) -> None:
        """Published watts == the engine's per-GPU watts x the layout's GPU count.

        Args:
            preset: A preset whose pick spans more than one GPU.
        """
        result = unwrap(coastline_recommender)(**COMMON_INPUTS, preset=preset)
        assert result["can_recommend"] is True
        total_gpus = result["total_gpus"]
        assert total_gpus > 1, (
            f"the {preset} preset picked {total_gpus} GPU(s); this guard only bites "
            f"on a multi-GPU layout, where the per-GPU and job-level figures differ"
        )

        per_gpu_watts = engine_per_gpu_watts(result, **COMMON_INPUTS)

        assert result["predicted_power_watts"] == pytest.approx(
            per_gpu_watts * total_gpus
        ), (
            f"predicted_power_watts must be the whole job's draw: the engine's "
            f"{per_gpu_watts:.2f} W per GPU across {total_gpus} GPUs = "
            f"{per_gpu_watts * total_gpus:.2f} W, not the {result['predicted_power_watts']:.2f} W "
            f"reported. Publishing the engine's per-GPU figure unscaled is the bug "
            f"this guards against."
        )

    @pytest.mark.parametrize("preset", MULTI_GPU_PRESETS)
    def test_tokens_per_watt_divides_the_two_published_figures(
        self, preset: str
    ) -> None:
        """Efficiency is throughput over power, both taken at the job level.

        Args:
            preset: A preset whose pick spans more than one GPU.
        """
        result = unwrap(coastline_recommender)(**COMMON_INPUTS, preset=preset)
        assert result["can_recommend"] is True
        total_gpus = result["total_gpus"]
        assert total_gpus > 1, (
            f"the {preset} preset picked {total_gpus} GPU(s); this guard only bites "
            f"on a multi-GPU layout, where the per-GPU and job-level figures differ"
        )

        assert result["tokens_per_watt"] == pytest.approx(
            result["predicted_throughput"] / result["predicted_power_watts"]
        ), "tokens_per_watt must be derived from the two figures actually published"

        per_gpu_watts = engine_per_gpu_watts(result, **COMMON_INPUTS)
        engine_figure = result["predicted_throughput"] / per_gpu_watts
        assert result["tokens_per_watt"] == pytest.approx(engine_figure / total_gpus), (
            f"the engine's own tokens_per_watt ({engine_figure:.2f}) divides a "
            f"cluster-total throughput by a per-GPU wattage and is inflated by "
            f"exactly total_gpus ({total_gpus}); the published figure must not be it"
        )

    @pytest.mark.parametrize("preset", MULTI_GPU_PRESETS)
    def test_power_is_plausible_for_the_whole_job(self, preset: str) -> None:
        """The draw sits where a busy multi-GPU job's draw physically must.

        A sanity bound with no engine arithmetic in it: whatever the model says, a
        fine-tuning job that keeps N GPUs busy draws a substantial fraction of those
        N boards' rated power, and cannot exceed it.

        Args:
            preset: A preset whose pick spans more than one GPU.
        """
        result = unwrap(coastline_recommender)(**COMMON_INPUTS, preset=preset)
        assert result["can_recommend"] is True
        low, high = job_power_bounds(SUPPORTED_GPU, result["total_gpus"])
        assert low <= result["predicted_power_watts"] <= high, (
            f"{result['predicted_power_watts']:.2f} W for {result['total_gpus']}x "
            f"{SUPPORTED_GPU} is outside [{low:.0f}, {high:.0f}] W: too low means a "
            f"per-GPU figure is being reported as the job's, too high means the "
            f"layout is being counted twice"
        )

    @pytest.mark.skipif(
        not autoconf_available(), reason="ado-autoconf is not installed"
    )
    def test_power_is_plausible_on_the_autoconf_path_too(self) -> None:
        """The README's worked example reports the job's draw, not one GPU's.

        The scaling lives in the plugin, not in a backend, so it has to hold whichever
        feasibility backend gated the grid.
        """
        result = unwrap(coastline_recommender)(**AUTOCONF_INPUTS, preset="balanced")
        assert result["can_recommend"] is True
        assert result["feasibility_backend"] == "autoconf"
        low, high = job_power_bounds(AUTOCONF_GPU, result["total_gpus"])
        assert low <= result["predicted_power_watts"] <= high, (
            f"{result['predicted_power_watts']:.2f} W for {result['total_gpus']}x "
            f"{AUTOCONF_GPU} is outside [{low:.0f}, {high:.0f}] W"
        )
        assert result["tokens_per_watt"] == pytest.approx(
            result["predicted_throughput"] / result["predicted_power_watts"]
        )
