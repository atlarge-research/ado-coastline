# Copyright 2026 Radu Nicolae
# SPDX-License-Identifier: Apache-2.0

"""COASTLINE recommender custom experiments for ado.

COASTLINE is "a better AutoConf that *uses* AutoConf": where the ``autoconf``
plugin answers "what is the minimum number of GPUs that will not OOM?", COASTLINE
reuses that very AutoConf feasibility check and then ranks the *feasible* layouts
by a performance/energy trade-off (Kavier analytical throughput + power model),
returning the best configuration under a chosen policy.

Two custom experiments are exposed:

``coastline_recommender``
    Multi-objective recommendation. A preset (``performance`` by default,
    ``balanced`` or ``energy``, and their ``-frontier`` variants) weights runtime
    against energy. This is the headline "better AutoConf" experiment.

``coastline_min_gpu_recommender``
    AutoConf's minimum-GPU algorithm, as in the autoconf ``min_gpu_recommender``: it
    keeps the job's total ``batch_size`` and returns the first GPU count in 1, 2, 4,
    ... (up to ``max_gpus``) that passes the feasibility check with the batch split
    evenly, and also reports predicted throughput and power for the pick.

Both default to AutoConf feasibility (``feasibility="autoconf"``) -- the engine's
structural guards (a positive GPU count, a per-device batch of at least 1) followed by
AutoConf's OOM classifier -- and degrade to the ``rules`` backend when AutoConf cannot
be loaded. ``rules`` here is those same structural guards *plus* COASTLINE's empirical
per-device token budget -- 60,224 tokens per device, fitted to the OOMs observed in
COASTLINE's calibration campaigns -- so a run without AutoConf still refuses the
per-device loads that were seen to run out of memory. See
:func:`_resolve_feasibility_backend`.

The COASTLINE engine itself is the optional ``coastline`` extra of this package (Python
3.11 to 3.13, the same range as this package); see :mod:`ado_coastline._bridge`. Without
it the experiments still register, and raise a typed error when measured.
"""

import logging
import math
import traceback
from typing import Any

from ado.modules.actuators.custom_experiments import custom_experiment
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.property import ConstitutiveProperty
from ado_coastline._bridge import CoastlineUnavailableError, import_facade

logger = logging.getLogger(__name__)

# Multi-objective presets understood by the COASTLINE selection layer.
_MULTI_OBJECTIVE_PRESETS = [
    "balanced",
    "energy",
    "performance",
    "balanced-frontier",
    "energy-frontier",
    "performance-frontier",
]

# The COASTLINE facade declares ``recommend(preset: str = "performance")``, so "no
# preset" has to be spelled as a string. ``min_gpu`` never reads it (see
# :func:`_preset_to_facade_args`).
_FACADE_DEFAULT_PRESET = "performance"


def _powers_of_two(limit: int) -> list[int]:
    """Enumerate the GPU budgets to search.

    Args:
        limit: The largest budget to consider.

    Returns:
        ``[1, 2, 4, ...]`` up to and including the largest power of two <= ``limit``.
    """
    result: list[int] = []
    budget = 1
    while budget <= limit:
        result.append(budget)
        budget *= 2
    return result


# Property definitions (shared by both experiments).

# Performance (Kavier) model. These are the model names COASTLINE's analytical
# throughput/power engine is calibrated for. AutoConf maps several of them onto
# its own vocabulary automatically (e.g. granite-3.3-8b -> granite-3.1-8b-instruct).
ModelName = ConstitutiveProperty(
    identifier="model_name",
    metadata={
        "description": "The LLM to fine-tune. Restricted to the models COASTLINE's "
        "analytical performance engine is calibrated for."
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=[
            "granite-3-8b",
            "granite-3.1-8b-instruct",
            "granite-3.3-8b",
            "llama3.2-3b",
            "mistral-7b-v0.1",
        ],
    ),
)

TuningMethod = ConstitutiveProperty(
    identifier="method",
    metadata={"description": "The fine-tuning method: full fine-tuning or LoRA."},
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["full", "lora"],
    ),
)

GPUModel = ConstitutiveProperty(
    identifier="gpu_model",
    metadata={"description": "The GPU model the job will run on."},
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=[
            "L40S",
            "NVIDIA-A100-80GB-PCIe",
            "NVIDIA-A100-SXM4-80GB",
            "NVIDIA-H100-PCIe",
        ],
    ),
)

TokensPerSample = ConstitutiveProperty(
    identifier="tokens_per_sample",
    metadata={"description": "Sequence length, in tokens, of one training sample."},
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],
    ),
)

BatchSize = ConstitutiveProperty(
    identifier="batch_size",
    metadata={
        "description": "Training batch size. For coastline_recommender it is the "
        "PER-DEVICE batch, held fixed while the GPU layout is optimised, so the job's "
        "total batch is batch_size x total_gpus. For coastline_min_gpu_recommender it "
        "is the job's TOTAL batch, as in autoconf's min_gpu_recommender: it is split "
        "evenly over the GPUs, so only GPU counts that divide it are tried. COASTLINE "
        "converts a per-device batch to AutoConf's total effective batch at the "
        "feasibility boundary."
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],
    ),
)

MaxGPUs = ConstitutiveProperty(
    identifier="max_gpus",
    metadata={
        "description": "Upper bound on the TOTAL number of GPUs a recommendation may "
        "use. The search considers the powers of two up to this value."
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],
    ),
)

GPUsPerNode = ConstitutiveProperty(
    identifier="gpus_per_node",
    metadata={
        "description": "GPUs available per node on the target cluster. Bounds the "
        "per-node width of the recommended layout (the 'gpus' output)."
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],
    ),
)

Preset = ConstitutiveProperty(
    identifier="preset",
    metadata={
        "description": "The multi-objective preset weighting runtime against energy: "
        "performance (default, 0.8/0.2), balanced (0.5/0.5) or energy (0.2/0.8). The "
        "'-frontier' variants select from the Pareto frontier."
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=list(_MULTI_OBJECTIVE_PRESETS),
    ),
)

Feasibility = ConstitutiveProperty(
    identifier="feasibility",
    metadata={
        "description": "How a candidate layout is checked: 'autoconf' (default) is the "
        "engine's structural guards (a positive GPU count, a per-device batch of at "
        "least 1) followed by AutoConf's OOM classifier, with no empirical budget on "
        "top; 'rules' is those same structural guards plus an empirical per-device "
        "token budget of 60,224 tokens per device, fitted to the OOMs observed in "
        "COASTLINE's calibration campaigns; 'none' is no check at all."
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["autoconf", "rules", "none"],
    ),
)

# Optional override for the AutoConf feasibility model name. Open categorical so
# any AutoConf-known model can be supplied; empty string -> fall back to model_name.
FeasibilityModel = ConstitutiveProperty(
    identifier="feasibility_model",
    metadata={
        "description": "Optional AutoConf model name for the feasibility check, for "
        "when the performance model is a proxy for the model actually being tuned. "
        "Empty string falls back to model_name."
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
        values=[""],
    ),
)

_OUTPUT_IDENTIFIERS = [
    "can_recommend",
    "gpus",
    "workers",
    "total_gpus",
    "recommended_batch_size",
    "predicted_throughput",
    "predicted_power_watts",
    # No `predicted_runtime_seconds` here, deliberately: runtime is dataset-size
    # dependent (runtime = total tokens / throughput) and this experiment declares
    # no dataset size at all -- no samples, epochs, steps or total tokens. Kavier
    # can compute a runtime, but only when it is given a job size, so there is
    # nothing to report and the field was always null. Do not re-add it without
    # first adding a dataset-size input property (e.g. `total_tokens`) and
    # threading it through `_run_recommendation`.
    "tokens_per_watt",
    "strategy",
    "feasibility_backend",
]


# Core recommendation logic (shared).


def _resolve_feasibility_backend(
    requested: str, autoconf_checker_cls: type, *, warn: bool = True
) -> str:
    """Resolve the feasibility backend, degrading ``autoconf`` to ``rules`` if needed.

    The whole point of COASTLINE is to *use* AutoConf, so ``autoconf`` is the
    default. When the engine's ``AutoconfFeasibilityChecker.available()`` probe says
    no, we degrade to the ``rules`` backend rather than failing outright, and signal
    that through the returned value.

    That probe is an *import* probe: it imports ado's ``autoconf`` modules and, with
    them, the AutoGluon stack they pull at module scope. So a missing ``ado-autoconf``
    is not what it normally catches -- ``ado-autoconf`` is a core dependency of
    ``coastline-recommender``, hence installed whenever the engine imports at all.
    What it catches is that import *raising*: a dependency skew in the AutoGluon/torch
    stack, a broken or incompatible install of one of them, or a missing transitive
    import. The engine logs the exception itself ("AutoConf import failed (will retry
    on next call)"), which is where the actual cause is; the plugin only sees a bool.
    Note what does *not* land here: the classifier's model artefacts are loaded lazily,
    at the first verdict, so artefacts that fail to load leave the backend on
    ``autoconf`` and turn every candidate infeasible instead of degrading the run.

    A degraded run is still memory-checked, just by a coarser mechanism. ``rules`` in
    this plugin means COASTLINE's ``RulesFeasibilityChecker`` -- a positive GPU count and
    a per-device batch of at least 1 -- *plus* the engine's empirical per-device token
    budget, which :func:`_run_recommendation` switches on for this backend (60,224 tokens
    per device, fitted to the OOMs observed in COASTLINE's calibration campaigns). So the
    fallback refuses the per-device loads that were seen to run out of memory, but it
    cannot make the model- and method-aware judgements AutoConf's classifier makes.

    Hence the warning below, and the ``feasibility_backend`` output property: a ``rules``
    row was gated by a different mechanism from an ``autoconf`` row, not by a weaker
    setting of the same one, and the two must be read separately.

    Resolving the backend here, and passing the result explicitly to
    ``coastline.Coastline(feasibility=...)``, is what keeps this side-effect free.
    COASTLINE only consults its ``COASTLINE_ALLOW_RULES_FALLBACK`` environment variable
    when it is asked for ``autoconf`` and cannot deliver it -- which cannot happen once
    the degrade has been applied here. Setting that variable instead would leak the
    permission to every later measurement sharing the Ray worker process.

    Args:
        requested: The requested backend: ``autoconf``, ``rules`` or ``none``.
        autoconf_checker_cls: COASTLINE's ``AutoconfFeasibilityChecker``, probed through
            its ``available()`` class method.
        warn: Whether to log the degradation. One measurement must report a degrade at
            most once. A measurement that gets as far as this call and then fails
            resolves the backend twice -- once for the attempt, once for
            :func:`_failure_result` to label the row -- so that second resolution
            passes ``False``. A measurement that fails *before* this call (the integer
            coercions at the top of :func:`_run_recommendation`) resolves it only in
            :func:`_failure_result`, and so logs no degradation at all even though its
            row still reads ``rules``.

    Returns:
        The backend to hand to the COASTLINE facade, and to report on the
        ``feasibility_backend`` output property.
    """
    if requested == "autoconf" and not autoconf_checker_cls.available():
        if warn:
            logger.warning(
                "feasibility='autoconf' requested but AutoConf is unavailable; "
                "degrading to the 'rules' backend. Nothing to install: 'ado-autoconf' "
                "is a core dependency of the COASTLINE engine, so it is already "
                "present whenever the engine imports. What failed is the engine's "
                "import probe -- a dependency skew in the AutoGluon/torch stack it "
                "pulls, a broken or incompatible install of one of them, or a missing "
                "transitive import. The engine logs the underlying exception as "
                "'AutoConf import failed (will retry on next call): ...'; read that "
                "line for the actual cause. This run is gated by COASTLINE's empirical "
                "per-device token budget (60,224 tokens per device, fitted to the OOMs "
                "observed in its calibration campaigns) instead of AutoConf's OOM "
                "classifier: a coarser check that knows nothing about the model or the "
                "tuning method. Read these rows separately from 'autoconf' ones."
            )
        return "rules"
    return requested


def _preset_to_facade_args(preset: str) -> tuple[str, str, str]:
    """Map a plugin preset onto the COASTLINE facade's strategy vocabulary.

    ``min_gpu`` has no preset: COASTLINE's ``PolicyFactory`` builds the min-GPU strategy
    without one, so the value is inert -- that policy selects on GPU count alone. The
    facade's signature is ``preset: str`` all the same, so pass its own default rather
    than ``None`` (checked: with ``strategy="min_gpu"`` the two return identical
    recommendations on every feasibility backend).

    Args:
        preset: A multi-objective preset name, or ``min_gpu``.

    Returns:
        ``(facade_strategy, facade_preset, strategy_label)``. The first two are passed
        to ``coastline.Coastline.recommend`` (the public facade resolves the selection
        policy and frontier normalization from the preset itself). The third is the
        value reported on the experiment's ``strategy`` output -- a stable
        ``coastline_<preset>`` / ``coastline_min_gpu`` label.
    """
    if preset == "min_gpu":
        return "min_gpu", _FACADE_DEFAULT_PRESET, "coastline_min_gpu"
    return "multi_objective", preset, f"coastline_{preset}"


def _run_recommendation(
    *,
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    max_gpus: int,
    gpus_per_node: int,
    preset: str,
    feasibility: str,
    feasibility_model: str,
) -> dict[str, Any]:
    """Build a COASTLINE workload/context and return the top recommendation.

    For a multi-objective preset, the per-device ``batch_size`` is held fixed;
    COASTLINE sweeps GPU layouts (powers of two up to ``max_gpus``), checks each for
    feasibility (AutoConf by default), simulates throughput + power with Kavier, and
    the preset's policy selects the winner. For ``min_gpu``, ``batch_size`` is the
    job's total batch: COASTLINE splits it evenly over 1, 2, 4, ... GPUs and returns
    the first feasible count. The work goes through the public ``coastline`` facade
    (``coastline.Coastline``), not COASTLINE pipeline internals.

    The ``rules`` backend -- requested outright or degraded to by
    :func:`_resolve_feasibility_backend` -- additionally enables the engine's empirical
    per-device token budget, so it is stricter here than the engine's own bare
    ``feasibility="rules"``. ``autoconf`` and ``none`` run without that guard.

    Args:
        model_name: The LLM to fine-tune (a Kavier-calibrated name).
        method: The fine-tuning method, ``full`` or ``lora``.
        gpu_model: The GPU model the job will run on.
        tokens_per_sample: Sequence length, in tokens, of one training sample.
        batch_size: For a multi-objective preset, the PER-DEVICE batch size, held
            fixed across the search; for ``min_gpu``, the job's TOTAL batch size.
            COASTLINE converts a per-device batch to AutoConf's total effective batch
            internally.
        max_gpus: Upper bound on the TOTAL GPUs a recommendation may use.
        gpus_per_node: GPUs available per node, bounding the layout's per-node width.
        preset: A multi-objective preset, or ``min_gpu``.
        feasibility: The requested feasibility backend.
        feasibility_model: Optional AutoConf model-name override; empty string to use
            ``model_name``.

    Returns:
        A dict keyed by ``_OUTPUT_IDENTIFIERS``.

    Raises:
        RuntimeError: If no feasible configuration exists (mapped to
            ``{"can_recommend": False}`` by :func:`_safe_recommend`).
        CoastlineUnavailableError: If the COASTLINE engine cannot be imported -- a
            genuine setup error, surfaced loudly.
    """
    Coastline, SystemContext, AutoconfFeasibilityChecker = import_facade()

    # ado's DISCRETE_VARIABLE_TYPE domains admit floats, and every one of these five
    # is an integer quantity downstream (GPU counts, a batch size, a sequence length).
    # Coerce them all, not just some: leaving one uncoerced is how a float leaks into
    # the engine and produces a plausible-looking but wrong prediction.
    max_gpus = int(max_gpus)
    gpus_per_node = int(gpus_per_node)
    batch_size = int(batch_size)
    tokens_per_sample = int(tokens_per_sample)

    backend = _resolve_feasibility_backend(feasibility, AutoconfFeasibilityChecker)
    facade_strategy, facade_preset, strategy_label = _preset_to_facade_args(preset)

    # Multi-objective: hold the per-device batch fixed and optimise the GPU layout
    # (powers of two). min_gpu: split the job's total batch over the fewest feasible GPUs.
    context = SystemContext.for_gpus(
        [gpu_model],
        max_gpus=max_gpus,
        gpus_per_node=gpus_per_node,
        max_nodes=max(1, math.ceil(max_gpus / max(gpus_per_node, 1))),
    )
    # Keys are WorkloadSpec field names -- COASTLINE's single input vocabulary. Its
    # `batch_size` is PER-DEVICE (Kavier's convention); with no layout given, `min_gpu`
    # reads it as the job's total batch on one GPU and splits it.
    workload: dict[str, Any] = {
        "llm_model": model_name,
        "fine_tuning_method": method,
        "gpu_model": gpu_model,
        "tokens_per_sample": tokens_per_sample,
        "batch_size": batch_size,
    }
    if feasibility_model:
        workload["feasibility_model"] = feasibility_model

    # The plugin's feasibility contract, in one expression. `autoconf`: the engine's
    # structural guards (a positive GPU count, a per-device batch of at least 1) and
    # then AutoConf's OOM classifier, with no empirical guard on top, so this path
    # stays identical to the runs COASTLINE's published results were produced from.
    # `rules` (asked for, or degraded to above): those same structural guards PLUS the
    # empirical per-device token budget, so a run without AutoConf still has a memory
    # check. `none`: nothing, truly.
    #
    # The budget bounds the per-device `batch_size x tokens_per_sample`. For a
    # multi-objective preset, `batch_sizes` below pins the grid to the one per-device
    # batch the caller asked for, so the budget vetoes the whole grid or none of it.
    # Under `min_gpu` the per-device batch is the total over the GPU count, so the
    # budget judges each count: an over-budget batch is split until it fits, and
    # refused only when no count up to `max_gpus` brings it under.
    engine = Coastline(
        predictor="kavier",
        energy="kavier_power",
        feasibility=backend,
        empirical_oom_guard=(backend == "rules"),
    )
    recommendations = engine.recommend(
        workload,
        context=context,
        strategy=facade_strategy,
        preset=facade_preset,
        # `total_gpus` is the TOTAL GPU budget per candidate, not a per-node count.
        total_gpus=_powers_of_two(max_gpus),
        batch_sizes=[batch_size],
        top_k=5,
        max_gpus=max_gpus,
    )
    if not recommendations:
        raise RuntimeError(
            "COASTLINE found no feasible configuration in the search space"
        )

    top = recommendations[0]
    meta = top.metadata or {}
    throughput = top.predicted_throughput or 0.0

    # UNIT BOUNDARY. COASTLINE reports `predicted_power_watts` as PER-GPU watts (its
    # power model is a single-GPU MSE fit; see the engine's KavierPowerPredictor, whose
    # docstring says "reports per-GPU watts"), while `predicted_throughput` is already
    # the CLUSTER TOTAL ("per-step tokens scale with TOTAL num_gpus" in the Kavier
    # engine). COASTLINE's own ranking multiplies up correctly -- its `_power_cost` is
    # `power * total_gpus` -- but the raw per-GPU figure is what reaches us here.
    #
    # This experiment publishes JOB-LEVEL quantities, so scale power to the whole job
    # before reporting it, and derive tokens_per_watt from the two job-level numbers
    # rather than trusting the engine's metadata (which divides a cluster-total
    # throughput by a per-GPU wattage and is therefore inflated by exactly total_gpus).
    power_per_gpu = meta.get("predicted_power_watts", 0.0) or 0.0
    power = power_per_gpu * top.total_gpus

    return {
        "can_recommend": True,
        # `gpus` is per node and `workers` is the node count, matching the autoconf
        # plugin's outputs; `total_gpus` is their product.
        "gpus": top.gpus_per_node,
        "workers": top.number_of_nodes,
        "total_gpus": top.total_gpus,
        "recommended_batch_size": meta.get("batch_size", batch_size),
        "predicted_throughput": throughput,
        "predicted_power_watts": power,
        "tokens_per_watt": throughput / power if power > 0 else 0.0,
        "strategy": strategy_label,
        "feasibility_backend": backend,
    }


def _failure_result(parameters: dict[str, Any]) -> dict[str, Any]:
    """Build the ``can_recommend=False`` payload for a run that produced nothing.

    Resolving the backend here rather than up front is deliberate: it needs the
    facade's AutoConf checker, and by the time a failure handler runs the engine is
    known to be importable (a missing engine is re-raised, never mapped to "no
    recommendation"). Reporting the backend on failures is what lets a sweep tell a
    candidate AutoConf vetoed from one the empirical token budget vetoed -- two
    different mechanisms, both reported as "no recommendation".

    This second resolution is silent (``warn=False``): the degrade belongs to the
    measurement, not to each function that asks which backend it ran, and the attempt
    that just failed has already logged it. The row still carries the degraded backend.

    Args:
        parameters: The keyword arguments the failed recommendation was called with.

    Returns:
        A dict with ``can_recommend=False`` and the feasibility backend that was in
        play, or ``can_recommend=False`` alone if the backend cannot be determined.
    """
    try:
        _, _, autoconf_checker_cls = import_facade()
        backend = _resolve_feasibility_backend(
            parameters.get("feasibility", "autoconf"), autoconf_checker_cls, warn=False
        )
    except Exception:  # noqa: BLE001 - reporting the backend must never mask the failure
        return {"can_recommend": False}
    return {"can_recommend": False, "feasibility_backend": backend}


def _safe_recommend(parameters: dict[str, Any]) -> dict[str, Any]:
    """Run a recommendation, mapping expected failures to ``can_recommend=False``.

    Args:
        parameters: Keyword arguments for :func:`_run_recommendation`.

    Returns:
        The recommendation dict, or a ``can_recommend=False`` dict when no feasible
        configuration exists or the inputs are invalid -- as the autoconf plugin does.
        The failure dict still carries ``feasibility_backend``, so a sweep can tell a
        candidate AutoConf rejected from one the empirical token budget rejected, and so
        the autoconf->rules degradation stays visible on rows that produced nothing.

    Raises:
        CoastlineUnavailableError: If the COASTLINE engine is missing. This is a setup
            error, so it is re-raised (ado turns it into InvalidMeasurements) rather
            than masked as "no recommendation".
    """
    try:
        return _run_recommendation(**parameters)
    except CoastlineUnavailableError:
        # Genuine environment/setup problem -- surface it, don't mask as "no rec".
        raise
    except RuntimeError as error:
        # Raised (here or by COASTLINE) when no feasible candidate exists. Note
        # CoastlineUnavailableError subclasses RuntimeError, so its re-raise above
        # must stay ahead of this handler.
        logger.warning(
            "COASTLINE produced no recommendation for %s: %s", parameters, error
        )
        return _failure_result(parameters)
    except ValueError as error:
        logger.warning(
            "COASTLINE recommendation invalid input for %s: %s", parameters, error
        )
        logger.debug("Traceback %s", traceback.format_exc())
        return _failure_result(parameters)


# Custom experiment 1: multi-objective recommender (headline).


@custom_experiment(
    required_properties=[
        ModelName,
        TuningMethod,
        GPUModel,
        TokensPerSample,
        BatchSize,
    ],
    optional_properties=[MaxGPUs, GPUsPerNode, Preset, Feasibility, FeasibilityModel],
    output_property_identifiers=_OUTPUT_IDENTIFIERS,
    metadata={
        "description": "COASTLINE multi-objective recommender: reuses the AutoConf "
        "feasibility check, then ranks feasible GPU layouts by a performance/energy "
        "trade-off (Kavier analytical throughput + power) under the chosen preset "
        "(performance by default, balanced, energy, and -frontier variants)."
    },
    parameterization={},
    version="1.0.0",
)
def coastline_recommender(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    max_gpus: int = 8,
    gpus_per_node: int = 8,
    preset: str = "performance",
    feasibility: str = "autoconf",
    feasibility_model: str = "",
) -> dict[str, Any]:
    """Recommend the best GPU layout for a tuning job (multi-objective).

    Args:
        model_name: The LLM to fine-tune.
        method: The fine-tuning method, ``full`` or ``lora``.
        gpu_model: The GPU model the job will run on.
        tokens_per_sample: Sequence length, in tokens, of one training sample.
        batch_size: PER-DEVICE batch size, held fixed across the search.
        max_gpus: Upper bound on the TOTAL GPUs a recommendation may use.
        gpus_per_node: GPUs available per node on the target cluster.
        preset: The multi-objective preset weighting runtime against energy.
        feasibility: The feasibility backend: ``autoconf``, ``rules`` or ``none``.
        feasibility_model: Optional AutoConf model-name override for the feasibility
            check; empty string to use ``model_name``.

    Returns:
        A dict keyed by ``_OUTPUT_IDENTIFIERS``, or ``{"can_recommend": False}`` when
        no feasible configuration was found.
    """
    return _safe_recommend(
        {
            "model_name": model_name,
            "method": method,
            "gpu_model": gpu_model,
            "tokens_per_sample": tokens_per_sample,
            "batch_size": batch_size,
            "max_gpus": max_gpus,
            "gpus_per_node": gpus_per_node,
            "preset": preset,
            "feasibility": feasibility,
            "feasibility_model": feasibility_model,
        }
    )


# Custom experiment 2: minimum-GPU recommender (AutoConf's algorithm).


@custom_experiment(
    required_properties=[
        ModelName,
        TuningMethod,
        GPUModel,
        TokensPerSample,
        BatchSize,
    ],
    optional_properties=[MaxGPUs, GPUsPerNode, Feasibility, FeasibilityModel],
    output_property_identifiers=_OUTPUT_IDENTIFIERS,
    metadata={
        "description": "COASTLINE minimum-GPU recommender, autoconf's "
        "min_gpu_recommender algorithm: it keeps the job's total batch_size and "
        "returns the first GPU count in 1, 2, 4, ... that is feasible with the batch "
        "split evenly, reported with Kavier predicted throughput and power."
    },
    parameterization={},
    version="1.0.0",
)
def coastline_min_gpu_recommender(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    max_gpus: int = 8,
    gpus_per_node: int = 8,
    feasibility: str = "autoconf",
    feasibility_model: str = "",
) -> dict[str, Any]:
    """Recommend the minimum feasible GPU layout for a tuning job.

    Args:
        model_name: The LLM to fine-tune.
        method: The fine-tuning method, ``full`` or ``lora``.
        gpu_model: The GPU model the job will run on.
        tokens_per_sample: Sequence length, in tokens, of one training sample.
        batch_size: The job's TOTAL batch size, kept and split evenly over the GPUs;
            only GPU counts that divide it are tried.
        max_gpus: Upper bound on the TOTAL GPUs a recommendation may use.
        gpus_per_node: GPUs available per node on the target cluster.
        feasibility: The feasibility backend: ``autoconf``, ``rules`` or ``none``.
        feasibility_model: Optional AutoConf model-name override for the feasibility
            check; empty string to use ``model_name``.

    Returns:
        A dict keyed by ``_OUTPUT_IDENTIFIERS``, or ``{"can_recommend": False}`` when
        no feasible configuration was found.
    """
    return _safe_recommend(
        {
            "model_name": model_name,
            "method": method,
            "gpu_model": gpu_model,
            "tokens_per_sample": tokens_per_sample,
            "batch_size": batch_size,
            "max_gpus": max_gpus,
            "gpus_per_node": gpus_per_node,
            "preset": "min_gpu",
            "feasibility": feasibility,
            "feasibility_model": feasibility_model,
        }
    )
