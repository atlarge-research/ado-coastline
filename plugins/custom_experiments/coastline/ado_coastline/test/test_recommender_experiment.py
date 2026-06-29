# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

"""Unit tests for the COASTLINE recommender custom experiments.

These tests bypass Ray/ado orchestration by calling each custom experiment's
``_original_func`` directly. They default to ``feasibility="rules"`` so they pass
without the heavy AutoConf (AutoGluon/torch) stack; one test exercises the
AutoConf path and is skipped when AutoConf is not installed.
"""

from collections.abc import Callable
from typing import Any

import pytest

from ado_coastline._bridge import coastline_available, find_coastline_root
from ado_coastline.recommender_experiment import (
    coastline_min_gpu_recommender,
    coastline_recommender,
)

# A Kavier-supported performance model used across the tests.
SUPPORTED_MODEL = "granite-3.3-8b"

coastline_missing = not coastline_available()
needs_coastline = pytest.mark.skipif(
    coastline_missing,
    reason="COASTLINE recommender not importable (pip install coastline-recommender "
    "or set COASTLINE_ROOT to a checkout)",
)


def _unwrap(experiment: Callable[..., Any]) -> Callable[..., Any]:
    """Return the undecorated function so we can call it without Ray."""
    return getattr(experiment, "_original_func", experiment)


def _autoconf_available() -> bool:
    if coastline_missing:
        return False
    try:
        from ado_coastline._bridge import import_facade

        _, _, autoconf_cls = import_facade()
        return bool(autoconf_cls.available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Bridge (always runs)
# ---------------------------------------------------------------------------


def test_coastline_is_importable() -> None:
    """The bridge can import COASTLINE (installed, or via the umbrella sibling fallback)."""
    assert coastline_available(), "Expected to import the COASTLINE recommender facade"


@pytest.mark.skipif(
    find_coastline_root() is None,
    reason="no sibling COASTLINE checkout (coastline is pip-installed instead)",
)
def test_find_coastline_root_locates_checkout() -> None:
    """When present, the sibling-checkout fallback resolves the COASTLINE layout."""
    root = find_coastline_root()
    assert root is not None, "Expected to locate the COASTLINE checkout"
    assert (root / "coastline" / "facade.py").is_file()
    assert (root / "coastline_recommender").is_dir()
    assert (root / "common" / "coastline_common").is_dir()


# ---------------------------------------------------------------------------
# Multi-objective recommender (rules feasibility -> no AutoGluon needed)
# ---------------------------------------------------------------------------


@needs_coastline
def test_balanced_recommendation_rules() -> None:
    """Balanced preset returns a usable GPU-layout recommendation."""
    result = _unwrap(coastline_recommender)(
        model_name=SUPPORTED_MODEL,
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=16,
        max_gpus=8,
        preset="balanced",
        feasibility="rules",
    )
    assert result["can_recommend"] is True
    assert result["gpus"] >= 1
    assert result["workers"] >= 1
    assert result["total_gpus"] == result["gpus"] * result["workers"]
    assert result["predicted_throughput"] > 0
    assert result["predicted_power_watts"] > 0
    assert result["feasibility_backend"] == "rules"
    assert result["strategy"].startswith("coastline_")


@needs_coastline
def test_output_contains_all_identifiers() -> None:
    """A successful recommendation reports every declared output identifier."""
    from ado_coastline.recommender_experiment import _OUTPUT_IDENTIFIERS

    result = _unwrap(coastline_recommender)(
        model_name=SUPPORTED_MODEL,
        method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=8,
        feasibility="rules",
    )
    assert result["can_recommend"] is True
    for key in _OUTPUT_IDENTIFIERS:
        assert key in result, f"missing output identifier: {key}"


@needs_coastline
@pytest.mark.parametrize("preset", ["balanced", "energy", "performance"])
def test_preset_variants_recommend(preset: str) -> None:
    """Each multi-objective preset produces a recommendation."""
    result = _unwrap(coastline_recommender)(
        model_name=SUPPORTED_MODEL,
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=16,
        preset=preset,
        feasibility="rules",
    )
    assert result["can_recommend"] is True
    assert result["strategy"] == f"coastline_{preset}"


@needs_coastline
def test_energy_preset_not_more_gpus_than_performance() -> None:
    """The energy preset should never pick *more* GPUs than the performance preset.

    Energy weights power (favouring fewer GPUs); performance weights speed. For a
    fixed batch the energy pick's total GPUs must be <= the performance pick's.
    """
    common = {
        "model_name": SUPPORTED_MODEL,
        "method": "full",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 2048,
        "batch_size": 16,
        "max_gpus": 8,
        "feasibility": "rules",
    }
    energy = _unwrap(coastline_recommender)(preset="energy", **common)
    performance = _unwrap(coastline_recommender)(preset="performance", **common)
    assert energy["can_recommend"] is True
    assert performance["can_recommend"] is True
    assert energy["total_gpus"] <= performance["total_gpus"]


@needs_coastline
def test_unsupported_model_cannot_recommend() -> None:
    """An unknown performance model yields no feasible candidate -> can_recommend False."""
    result = _unwrap(coastline_recommender)(
        model_name="totally-unknown-model-xyz",
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=16,
        feasibility="rules",
    )
    assert result["can_recommend"] is False


# ---------------------------------------------------------------------------
# Minimum-GPU recommender
# ---------------------------------------------------------------------------


@needs_coastline
def test_min_gpu_recommender_rules() -> None:
    """min_gpu recommender returns the smallest feasible layout with predictions."""
    result = _unwrap(coastline_min_gpu_recommender)(
        model_name=SUPPORTED_MODEL,
        method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=2048,
        batch_size=16,
        max_gpus=8,
        feasibility="rules",
    )
    assert result["can_recommend"] is True
    assert result["total_gpus"] >= 1
    assert result["strategy"] == "coastline_min_gpu"
    assert result["predicted_throughput"] > 0


@needs_coastline
def test_min_gpu_not_more_than_balanced() -> None:
    """min_gpu must use <= the GPUs of the balanced multi-objective pick."""
    common = {
        "model_name": SUPPORTED_MODEL,
        "method": "full",
        "gpu_model": "NVIDIA-A100-SXM4-80GB",
        "tokens_per_sample": 2048,
        "batch_size": 16,
        "max_gpus": 8,
        "feasibility": "rules",
    }
    min_gpu = _unwrap(coastline_min_gpu_recommender)(**common)
    balanced = _unwrap(coastline_recommender)(preset="balanced", **common)
    assert min_gpu["total_gpus"] <= balanced["total_gpus"]


# ---------------------------------------------------------------------------
# AutoConf path (the "uses AutoConf" claim) -- skipped if AutoConf is absent
# ---------------------------------------------------------------------------


@needs_coastline
@pytest.mark.skipif(
    not _autoconf_available(), reason="AutoConf (ado-autoconf) not installed"
)
def test_autoconf_feasibility_path() -> None:
    """With AutoConf installed, the default feasibility backend gates the grid."""
    result = _unwrap(coastline_recommender)(
        model_name=SUPPORTED_MODEL,
        method="lora",
        gpu_model="NVIDIA-A100-80GB-PCIe",
        tokens_per_sample=2048,
        batch_size=16,
        max_gpus=8,
        preset="balanced",
        feasibility="autoconf",
    )
    assert result["can_recommend"] is True
    assert result["feasibility_backend"] == "autoconf"
    assert result["total_gpus"] >= 1
