# Copyright 2026 Radu Nicolae
# SPDX-License-Identifier: Apache-2.0

"""Registration and contract tests for the COASTLINE custom experiments.

These tests never drive the COASTLINE recommender engine, so they run everywhere ado
runs -- including CI, where the optional ``coastline`` extra is not installed because
it pulls ado-autoconf and with it the autogluon/torch stack. They assert the part of
the plugin that is always live: importing
:mod:`ado_coastline.recommender_experiment` registers both experiments in ado's
custom-experiment catalog with the inputs, outputs, defaults and version that the
plugin README documents, and :mod:`ado_coastline._bridge` turns an engine that cannot
be imported into one typed, actionable error.
"""

import importlib.metadata
import re
import sys
import types
from collections.abc import Callable
from typing import Any, NoReturn

import pytest
from ado_coastline._bridge import (
    INSTALL_HINT,
    CoastlineUnavailableError,
    coastline_available,
    import_facade,
)
from ado_coastline.recommender_experiment import (
    coastline_min_gpu_recommender,
    coastline_recommender,
)

from ado.modules.actuators.custom_experiments import get_custom_experiments_catalog
from ado.schema.domain import VariableTypeEnum
from ado.schema.experiment import Experiment
from ado.schema.reference import ExperimentReference

#: Distribution name of this plugin, used to check its declared entry point.
DISTRIBUTION_NAME = "ado-coastline"

#: The ado actuator both experiments belong to.
ACTUATOR_IDENTIFIER = "custom_experiments"

#: Plugin VERSION, which both ``@custom_experiment`` decorators must declare.
EXPERIMENT_VERSION = "1.0.0"

#: Both experiment identifiers, as registered in the catalog.
EXPERIMENT_IDENTIFIERS = [
    "coastline_recommender",
    "coastline_min_gpu_recommender",
]

#: The outputs documented in the README "Outputs" section, in declaration order.
README_OUTPUT_PROPERTIES = [
    "can_recommend",
    "gpus",
    "workers",
    "total_gpus",
    "recommended_batch_size",
    "predicted_throughput",
    "predicted_power_watts",
    "tokens_per_watt",
    "strategy",
    "feasibility_backend",
]

#: The inputs the README marks "Required: yes", shared by both experiments.
README_REQUIRED_PROPERTIES = [
    "model_name",
    "method",
    "gpu_model",
    "tokens_per_sample",
    "batch_size",
]

#: The optional inputs and their README defaults, per experiment. ``preset`` is
#: documented as belonging to ``coastline_recommender`` only.
README_OPTIONAL_DEFAULTS = {
    "coastline_recommender": {
        "max_gpus": 8,
        "gpus_per_node": 8,
        "preset": "performance",
        "feasibility": "autoconf",
        "feasibility_model": "",
    },
    "coastline_min_gpu_recommender": {
        "max_gpus": 8,
        "gpus_per_node": 8,
        "feasibility": "autoconf",
        "feasibility_model": "",
    },
}

#: A point inside every declared input domain, so ado's entity validation passes
#: and the call reaches the engine boundary.
VALID_INPUTS = {
    "model_name": "granite-3.3-8b",
    "method": "full",
    "gpu_model": "NVIDIA-A100-SXM4-80GB",
    "tokens_per_sample": 2048,
    "batch_size": 16,
}

#: The message a dependency skew inside the engine surfaces with. It is deliberately
#: not an import-flavoured string: that is the whole point of the skew tests below.
SKEW_ERROR_MESSAGE = "pydantic model rebuild failed inside coastline"

#: Resolved once: importing the engine is the expensive part of the probe.
ENGINE_AVAILABLE = coastline_available()

try:
    PLUGIN_DISTRIBUTION: importlib.metadata.Distribution | None = (
        importlib.metadata.distribution(DISTRIBUTION_NAME)
    )
except importlib.metadata.PackageNotFoundError:
    PLUGIN_DISTRIBUTION = None


def catalog_experiment(identifier: str) -> Experiment:
    """Return the uniquely registered catalog entry for an experiment identifier.

    Args:
        identifier: The base experiment identifier to look up.

    Returns:
        The single :class:`~ado.schema.experiment.Experiment` registered under
        ``identifier`` in ado's custom-experiment catalog.
    """
    matches = [
        experiment
        for experiment in get_custom_experiments_catalog().experiments
        if experiment.identifier == identifier
    ]
    assert len(matches) == 1, (
        f"expected exactly one catalog entry for {identifier!r}, found {len(matches)}"
    )
    return matches[0]


def decorated_experiment(identifier: str) -> Callable[..., Any]:
    """Return the decorated experiment function for an experiment identifier.

    Args:
        identifier: The base experiment identifier to look up.

    Returns:
        The ``@custom_experiment``-decorated function of that name.
    """
    return {
        "coastline_recommender": coastline_recommender,
        "coastline_min_gpu_recommender": coastline_min_gpu_recommender,
    }[identifier]


class ExplodingCoastlineModule(types.ModuleType):
    """A stand-in for a ``coastline`` package that is installed but unusable.

    The engine keeps a bare ``import coastline`` cheap by resolving its public names
    through a PEP 562 module ``__getattr__`` that imports the heavy subpackages on
    first access. A dependency skew therefore runs third-party module bodies *during*
    the bridge's ``from coastline import ...`` and raises whatever they raise, at the
    attribute lookup rather than at the import statement.

    ``TypeError`` stands in for that whole class of failure. It must not be an
    ``AttributeError``: the import machinery swallows those while it tries the
    submodule fallback and re-raises ``ImportError``, which is exactly the case the
    bridge always handled.
    """

    def __getattr__(self, name: str) -> NoReturn:
        """Fail the way a dependency skew inside the engine fails.

        Args:
            name: The attribute the bridge asked this module for.

        Raises:
            TypeError: Always -- that is the point of the stand-in.
        """
        raise TypeError(SKEW_ERROR_MESSAGE)


class TestCatalogRegistration:
    """Importing the module must register both experiments with ado."""

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_experiment_is_registered(self, identifier: str) -> None:
        """Both experiments appear in ado's custom-experiment catalog on import.

        Args:
            identifier: The experiment identifier under test.
        """
        registered = {
            experiment.identifier
            for experiment in get_custom_experiments_catalog().experiments
        }
        assert identifier in registered

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_experiment_is_not_deprecated(self, identifier: str) -> None:
        """Neither experiment is registered as deprecated.

        Args:
            identifier: The experiment identifier under test.
        """
        assert catalog_experiment(identifier).deprecated is False

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_actuator_identifier(self, identifier: str) -> None:
        """Both experiments are owned by the ``custom_experiments`` actuator.

        Args:
            identifier: The experiment identifier under test.
        """
        assert catalog_experiment(identifier).actuatorIdentifier == ACTUATOR_IDENTIFIER

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_experiment_version(self, identifier: str) -> None:
        """Both experiments declare the plugin VERSION, ``1.0.0``.

        Args:
            identifier: The experiment identifier under test.
        """
        assert catalog_experiment(identifier).version == EXPERIMENT_VERSION

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_major_version_identifier(self, identifier: str) -> None:
        """A 1.0.0 experiment is keyed in the catalog as ``<identifier>@v1``.

        Args:
            identifier: The experiment identifier under test.
        """
        assert catalog_experiment(identifier).major_version_identifier == (
            f"{identifier}@v1"
        )

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_resolves_from_fully_qualified_reference(self, identifier: str) -> None:
        """A reference pinned to 1.0.0 resolves to the registered experiment.

        Args:
            identifier: The experiment identifier under test.
        """
        reference = ExperimentReference(
            experimentIdentifier=identifier,
            actuatorIdentifier=ACTUATOR_IDENTIFIER,
            experimentVersion=EXPERIMENT_VERSION,
        )
        resolved = get_custom_experiments_catalog().experimentForReference(
            reference,
            match_on="fully_qualified_version",
            resolve=True,
        )
        assert resolved is not None
        assert resolved.identifier == identifier

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_decorated_function_carries_the_catalog_entry(
        self, identifier: str
    ) -> None:
        """The decorated function exposes the same Experiment that was registered.

        Args:
            identifier: The experiment identifier under test.
        """
        function = decorated_experiment(identifier)
        assert function._is_custom_experiment is True
        assert function._experiment.model_dump() == (
            catalog_experiment(identifier).model_dump()
        )

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_experiment_round_trips_through_a_dump(self, identifier: str) -> None:
        """create -> dump -> create from dump preserves the experiment.

        Args:
            identifier: The experiment identifier under test.
        """
        experiment = catalog_experiment(identifier)
        dumped = experiment.model_dump()
        assert Experiment.model_validate(dumped).model_dump() == dumped

    @pytest.mark.skipif(
        PLUGIN_DISTRIBUTION is None,
        reason=f"{DISTRIBUTION_NAME} is not installed, so it declares no entry point",
    )
    def test_entry_point_targets_the_experiment_module(self) -> None:
        """The installed distribution advertises the experiments to ado."""
        assert PLUGIN_DISTRIBUTION is not None
        declared = {
            entry_point.name: entry_point.value
            for entry_point in PLUGIN_DISTRIBUTION.entry_points
            if entry_point.group == "ado.custom_experiments"
        }
        assert declared == {"coastline": "ado_coastline.recommender_experiment"}


class TestReadmeContract:
    """The registered experiments must match what the README promises."""

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_output_property_identifiers(self, identifier: str) -> None:
        """Both experiments declare exactly the README's output properties.

        Args:
            identifier: The experiment identifier under test.
        """
        declared = [
            target.identifier
            for target in catalog_experiment(identifier).targetProperties
        ]
        assert declared == README_OUTPUT_PROPERTIES

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_required_properties(self, identifier: str) -> None:
        """Both experiments require exactly the README's required inputs.

        Args:
            identifier: The experiment identifier under test.
        """
        declared = [
            prop.identifier
            for prop in catalog_experiment(identifier).requiredProperties
        ]
        assert declared == README_REQUIRED_PROPERTIES

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_optional_properties(self, identifier: str) -> None:
        """Only ``coastline_recommender`` takes a ``preset``, per the README.

        Args:
            identifier: The experiment identifier under test.
        """
        declared = {
            prop.identifier
            for prop in catalog_experiment(identifier).optionalProperties
        }
        assert declared == set(README_OPTIONAL_DEFAULTS[identifier])

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_optional_property_defaults(self, identifier: str) -> None:
        """The default parameterization matches the README's default column.

        Args:
            identifier: The experiment identifier under test.
        """
        defaults = {
            value.property.identifier: value.value
            for value in catalog_experiment(identifier).defaultParameterization
        }
        assert defaults == README_OPTIONAL_DEFAULTS[identifier]

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_model_name_domain_is_the_documented_catalog(self, identifier: str) -> None:
        """``model_name`` is categorical over the Kavier-calibrated models.

        Args:
            identifier: The experiment identifier under test.
        """
        experiment = catalog_experiment(identifier)
        model_name = next(
            prop
            for prop in experiment.requiredProperties
            if prop.identifier == "model_name"
        )
        assert model_name.propertyDomain.variableType == (
            VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
        )
        assert model_name.propertyDomain.values == [
            "granite-3-8b",
            "granite-3.1-8b-instruct",
            "granite-3.3-8b",
            "llama3.2-3b",
            "mistral-7b-v0.1",
        ]

    def test_preset_domain_lists_every_documented_preset(self) -> None:
        """``preset`` offers the three policies and their frontier variants."""
        experiment = catalog_experiment("coastline_recommender")
        preset = next(
            prop
            for prop in experiment.optionalProperties
            if prop.identifier == "preset"
        )
        assert set(preset.propertyDomain.values) == {
            "balanced",
            "energy",
            "performance",
            "balanced-frontier",
            "energy-frontier",
            "performance-frontier",
        }

    def test_feasibility_domain_lists_every_documented_backend(self) -> None:
        """``feasibility`` offers ``autoconf``, ``rules`` and ``none``."""
        experiment = catalog_experiment("coastline_recommender")
        feasibility = next(
            prop
            for prop in experiment.optionalProperties
            if prop.identifier == "feasibility"
        )
        assert set(feasibility.propertyDomain.values) == {"autoconf", "rules", "none"}


class TestInputValidation:
    """ado validates inputs against the declared domains before any engine work."""

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_out_of_domain_model_name_is_rejected(self, identifier: str) -> None:
        """An unknown ``model_name`` fails validation, engine or no engine.

        Args:
            identifier: The experiment identifier under test.
        """
        arguments = {**VALID_INPUTS, "model_name": "totally-unknown-model-xyz"}
        with pytest.raises(
            ValueError, match="do not match required/optional properties"
        ):
            decorated_experiment(identifier)(**arguments)

    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_out_of_domain_gpu_model_is_rejected(self, identifier: str) -> None:
        """An unknown ``gpu_model`` fails validation, engine or no engine.

        Args:
            identifier: The experiment identifier under test.
        """
        arguments = {**VALID_INPUTS, "gpu_model": "NVIDIA-GeForce-RTX-4090"}
        with pytest.raises(
            ValueError, match="do not match required/optional properties"
        ):
            decorated_experiment(identifier)(**arguments)


class TestEngineBoundary:
    """The bridge's availability probe and its typed unusable-engine error."""

    def test_availability_probe_agrees_with_import_facade(self) -> None:
        """``coastline_available()`` reports exactly what ``import_facade()`` does."""
        if ENGINE_AVAILABLE:
            assert len(import_facade()) == 3
        else:
            with pytest.raises(CoastlineUnavailableError, match=r"(?i)coastline"):
                import_facade()

    def test_dependency_skew_is_reported_as_the_typed_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An engine that is installed but broken raises the typed error too.

        Regression guard. The bridge once trapped only ``ImportError``, which covers
        a *missing* engine but not an unusable one: because the engine resolves its
        submodules lazily, a dependency skew inside it raises a third-party exception
        that escaped raw through every experiment call, losing both the install hint
        and the promise that the plugin has exactly one failure mode at this boundary.

        Args:
            monkeypatch: Swaps in a ``coastline`` package that fails on attribute
                access, restoring the real one afterwards.
        """
        monkeypatch.setitem(
            sys.modules, "coastline", ExplodingCoastlineModule("coastline")
        )
        with pytest.raises(CoastlineUnavailableError) as raised:
            import_facade()
        assert INSTALL_HINT in str(raised.value), (
            "the typed error must keep telling the user how to install the engine"
        )
        assert SKEW_ERROR_MESSAGE in str(raised.value), (
            "the typed error must carry the original failure, not just replace it"
        )
        assert isinstance(raised.value.__cause__, TypeError), (
            "the original exception must survive as __cause__ for the traceback"
        )

    def test_dependency_skew_makes_the_availability_probe_answer_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``coastline_available()`` answers a broken engine with False, not a raise.

        The probe is called at import time by this module and by the engine-dependent
        suite, so a third-party exception escaping it would break collection rather
        than skip a test.

        Args:
            monkeypatch: Swaps in a ``coastline`` package that fails on attribute
                access, restoring the real one afterwards.
        """
        monkeypatch.setitem(
            sys.modules, "coastline", ExplodingCoastlineModule("coastline")
        )
        assert coastline_available() is False

    @pytest.mark.skipif(
        ENGINE_AVAILABLE,
        reason="the 'coastline' extra is installed, so the engine is importable",
    )
    @pytest.mark.parametrize("identifier", EXPERIMENT_IDENTIFIERS)
    def test_measurement_without_the_engine_raises_typed_error(
        self, identifier: str
    ) -> None:
        """Without the extra, a measurement fails loudly rather than silently.

        A missing engine is a setup error, not a "no recommendation" answer, so it
        must not be mapped onto ``can_recommend=False``. The error must also carry
        the install hint.

        Args:
            identifier: The experiment identifier under test.
        """
        with pytest.raises(CoastlineUnavailableError, match=re.escape(INSTALL_HINT)):
            decorated_experiment(identifier)(**VALID_INPUTS)
