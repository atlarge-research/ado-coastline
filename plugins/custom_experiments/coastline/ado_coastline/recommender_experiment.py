# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

"""COASTLINE recommender custom experiments for ado.

COASTLINE is "a better AutoConf that *uses* AutoConf": where the ``autoconf``
plugin answers "what is the minimum number of GPUs that will not OOM?", COASTLINE
reuses that very AutoConf feasibility check and then ranks the *feasible* layouts
by a performance/energy trade-off (Kavier analytical throughput + power model),
returning the best configuration under a chosen policy.

Two custom experiments are exposed:

``coastline_recommender``
    Multi-objective recommendation. A preset (``balanced`` / ``energy`` /
    ``performance`` and their ``-frontier`` variants) weights the runtime-vs-power
    trade-off. This is the headline "better AutoConf" experiment.

``coastline_min_gpu_recommender``
    The minimum-GPU policy over the same AutoConf-gated feasible set -- a direct,
    COASTLINE-driven analogue of the autoconf ``min_gpu_recommender`` that also
    reports predicted throughput and power for the pick.

Both default to AutoConf feasibility (``feasibility="autoconf"``) and gracefully
degrade to a divisibility rule when AutoConf is not installed.
"""

from __future__ import annotations

import logging
import math
import os
import traceback
from typing import Any

from ado_coastline._bridge import CoastlineUnavailableError, import_facade
from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

moduleLog = logging.getLogger(__name__)

# AutoConf model version used for the feasibility (OOM) classifier.
AUTOCONF_MODEL_VERSION = "3.1.0"

# Multi-objective presets understood by the COASTLINE selection layer.
_MULTI_OBJECTIVE_PRESETS = [
    "balanced",
    "energy",
    "performance",
    "balanced-frontier",
    "energy-frontier",
    "performance-frontier",
]


def _powers_of_two(limit: int) -> list[int]:
    """Return [1, 2, 4, ...] up to and including the largest power of 2 <= limit."""
    result: list[int] = []
    g = 1
    while g <= limit:
        result.append(g)
        g *= 2
    return result


# ---------------------------------------------------------------------------
# Property definitions (shared by both experiments)
# ---------------------------------------------------------------------------

# Performance (Kavier) model. These are the model names COASTLINE's analytical
# throughput/power engine is calibrated for. AutoConf maps several of them onto
# its own vocabulary automatically (e.g. granite-3.3-8b -> granite-3.1-8b-instruct).
ModelName = ConstitutiveProperty(
    identifier="model_name",
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
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["full", "lora"],
    ),
)

GPUModel = ConstitutiveProperty(
    identifier="gpu_model",
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
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],
    ),
)

BatchSize = ConstitutiveProperty(
    identifier="batch_size",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],
    ),
)

MaxGPUs = ConstitutiveProperty(
    identifier="max_gpus",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],
    ),
)

GPUsPerNode = ConstitutiveProperty(
    identifier="gpus_per_node",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 10000001],
    ),
)

Preset = ConstitutiveProperty(
    identifier="preset",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=list(_MULTI_OBJECTIVE_PRESETS),
    ),
)

Feasibility = ConstitutiveProperty(
    identifier="feasibility",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["autoconf", "rules", "none"],
    ),
)

# Optional override for the AutoConf feasibility model name. Open categorical so
# any AutoConf-known model can be supplied; empty string -> fall back to model_name.
FeasibilityModel = ConstitutiveProperty(
    identifier="feasibility_model",
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
    "predicted_runtime_seconds",
    "tokens_per_watt",
    "strategy",
    "feasibility_backend",
]


# ---------------------------------------------------------------------------
# Core recommendation logic (shared)
# ---------------------------------------------------------------------------


def _resolve_feasibility_backend(requested: str, autoconf_checker_cls: type) -> str:
    """Return the feasibility backend to use, degrading autoconf->rules if needed.

    The whole point of COASTLINE is to *use* AutoConf, so ``autoconf`` is the
    default. When the AutoConf package/model cannot be loaded (e.g. ``ado-autoconf``
    not installed in this environment) we fall back to the divisibility rule rather
    than failing outright, and signal this via the returned value.
    """
    if requested == "autoconf" and not autoconf_checker_cls.available():
        moduleLog.warning(
            "feasibility='autoconf' requested but AutoConf is unavailable "
            "(install 'ado-autoconf'); degrading to divisibility 'rules'."
        )
        # Allow COASTLINE's own factory to build the rules checker without raising.
        os.environ["COASTLINE_ALLOW_RULES_FALLBACK"] = "1"
        return "rules"
    return requested


def _preset_to_facade_args(preset: str) -> tuple[str, str | None, str]:
    """Map a plugin preset to ``(facade_strategy, facade_preset, strategy_label)``.

    ``facade_strategy`` / ``facade_preset`` are passed to ``coastline.Coastline.recommend``
    (the public facade resolves the selection policy and frontier normalization from the
    preset itself). ``strategy_label`` is the value reported back on the experiment's
    ``strategy`` output -- a stable ``coastline_<preset>`` / ``coastline_min_gpu`` label.
    """
    if preset == "min_gpu":
        return "min_gpu", None, "coastline_min_gpu"
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

    The user-supplied ``batch_size`` is held fixed; COASTLINE sweeps GPU layouts
    (powers of two up to ``max_gpus``), checks each for feasibility (AutoConf by
    default), simulates throughput + power with Kavier, and the policy implied by
    ``preset`` selects the winner. The work goes through the public ``coastline``
    facade (``coastline.Coastline``), not COASTLINE pipeline internals.

    Returns a dict keyed by ``_OUTPUT_IDENTIFIERS``. Raises ``RuntimeError`` when no
    feasible configuration exists (mapped to ``{"can_recommend": False}`` upstream) and
    :class:`CoastlineUnavailableError` if the COASTLINE engine cannot be imported (a
    genuine setup error, surfaced loudly).
    """
    Coastline, SystemContext, AutoconfFeasibilityChecker = import_facade()

    max_gpus = int(max_gpus)
    gpus_per_node = int(gpus_per_node)
    batch_size = int(batch_size)

    backend = _resolve_feasibility_backend(feasibility, AutoconfFeasibilityChecker)
    facade_strategy, facade_preset, strategy_label = _preset_to_facade_args(preset)

    # Hold the user's batch size fixed; optimise the GPU layout (powers of two) only.
    context = SystemContext.for_gpus(
        [gpu_model],
        max_gpus=max_gpus,
        gpus_per_node=gpus_per_node,
        max_nodes=max(1, math.ceil(max_gpus / max(gpus_per_node, 1))),
    )
    workload: dict[str, Any] = {
        "llm_model": model_name,
        "fine_tuning_method": method,
        "gpu_model": gpu_model,
        "tokens_per_sample": tokens_per_sample,
        "batch_size": batch_size,
    }
    if feasibility_model:
        workload["feasibility_model"] = feasibility_model

    engine = Coastline(
        throughput_estim="kavier", energy="kavier_power", feasibility=backend
    )
    recommendations = engine.recommend(
        workload,
        context=context,
        strategy=facade_strategy,
        preset=facade_preset,
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
    power = meta.get("predicted_power_watts", 0.0) or 0.0
    throughput = top.predicted_throughput or 0.0

    return {
        "can_recommend": True,
        "gpus": top.gpus_per_node,
        "workers": top.number_of_nodes,
        "total_gpus": top.total_gpus,
        "recommended_batch_size": meta.get("batch_size", batch_size),
        "predicted_throughput": throughput,
        "predicted_power_watts": power,
        "predicted_runtime_seconds": top.predicted_runtime_seconds,
        "tokens_per_watt": meta.get(
            "tokens_per_watt", (throughput / power if power > 0 else 0.0)
        ),
        "strategy": strategy_label,
        "feasibility_backend": backend,
    }


def _safe_recommend(parameters: dict[str, Any]) -> dict[str, Any]:
    """Run a recommendation, mapping expected failures to ``can_recommend=False``.

    A missing COASTLINE engine is re-raised (setup error -> ado InvalidMeasurements);
    "no feasible configuration" and input-validation failures return
    ``{"can_recommend": False}`` as the autoconf plugin does.
    """
    try:
        return _run_recommendation(**parameters)
    except CoastlineUnavailableError:
        # Genuine environment/setup problem -- surface it, don't mask as "no rec".
        raise
    except RuntimeError as exc:
        # Raised (here or by COASTLINE) when no feasible candidate exists. Note
        # CoastlineUnavailableError subclasses RuntimeError, so its re-raise above
        # must stay ahead of this handler.
        moduleLog.warning(
            "COASTLINE produced no recommendation for %s: %s", parameters, exc
        )
        return {"can_recommend": False}
    except ValueError as exc:
        moduleLog.warning(
            "COASTLINE recommendation invalid input for %s: %s", parameters, exc
        )
        moduleLog.debug("Traceback %s", traceback.format_exc())
        return {"can_recommend": False}


# ---------------------------------------------------------------------------
# Custom experiment 1: multi-objective recommender (headline)
# ---------------------------------------------------------------------------


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
        "(balanced/energy/performance, and -frontier variants)."
    },
    parameterization={},
)
def coastline_recommender(
    model_name: str,
    method: str,
    gpu_model: str,
    tokens_per_sample: int,
    batch_size: int,
    max_gpus: int = 8,
    gpus_per_node: int = 8,
    preset: str = "balanced",
    feasibility: str = "autoconf",
    feasibility_model: str = "",
) -> dict[str, Any]:
    """Recommend the best GPU layout for a tuning job (multi-objective)."""
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


# ---------------------------------------------------------------------------
# Custom experiment 2: minimum-GPU recommender (AutoConf analogue)
# ---------------------------------------------------------------------------


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
        "description": "COASTLINE minimum-GPU recommender: the smallest feasible GPU "
        "layout over the AutoConf-gated feasible set, reported with Kavier predicted "
        "throughput and power. A COASTLINE-driven analogue of autoconf's "
        "min_gpu_recommender."
    },
    parameterization={},
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
    """Recommend the minimum feasible GPU layout for a tuning job."""
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
