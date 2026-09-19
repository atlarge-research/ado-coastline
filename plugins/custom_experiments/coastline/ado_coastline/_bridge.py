# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Lazy import of the COASTLINE recommender engine.

COASTLINE is published as ``coastline-recommender`` and installs exactly one importable
package, ``coastline``, whose public facade (``coastline.Coastline``) is what this plugin
drives -- never COASTLINE pipeline internals.

The engine pulls ado-autoconf and therefore the autogluon/torch stack, so this plugin
declares it as the optional ``coastline`` extra rather than a hard dependency: the plugin
installs and registers its experiments anywhere ado runs, and only an actual *measurement*
needs the engine. The
import therefore happens lazily, inside the experiment call rather than at module import,
and an unusable engine surfaces as a typed :class:`CoastlineUnavailableError` carrying
installation guidance instead of some raw third-party failure at plugin-registration time.
"""

INSTALL_HINT = (
    "The COASTLINE recommender engine is not installed. Install it with: "
    'pip install "ado-coastline[coastline]"  (from an ado checkout: '
    'uv pip install -e "plugins/custom_experiments/coastline[coastline]")'
)


class CoastlineUnavailableError(RuntimeError):
    """Raised when the COASTLINE recommender engine cannot be imported."""


def import_facade() -> tuple[type, type, type]:
    """Import the COASTLINE symbols this plugin needs.

    Returns:
        The tuple ``(Coastline, SystemContext, AutoconfFeasibilityChecker)``. The first
        two drive the public facade; the third is used only as an availability probe
        (``AutoconfFeasibilityChecker.available()``) so the plugin can report which
        feasibility backend it actually used.

    Raises:
        CoastlineUnavailableError: If the engine cannot be imported, for *any* reason.
            "Not installed" is only the common case. ``coastline`` keeps a bare
            ``import coastline`` cheap by resolving its public names lazily (a PEP 562
            module ``__getattr__`` that imports ``coastline.sdk.recommend``, and with it
            pandas/kavier, on first attribute access), so a dependency skew inside the
            engine -- a pydantic or pandas mismatch, say -- executes third-party module
            bodies *during* these statements and surfaces as whatever they raise: a
            ``TypeError``, a ``RuntimeError``, a validation error of someone else's
            design. None of those are ``ImportError``, and an unusable engine is an
            unusable engine however it fails, so the trap is deliberately broad. The
            original exception is kept both as the message tail and as ``__cause__``.
    """
    try:
        from coastline import Coastline
        from coastline.sdk.models.context import SystemContext
        from coastline.sdk.predictors.feasibility.autoconf import (
            AutoconfFeasibilityChecker,
        )
    except Exception as error:
        raise CoastlineUnavailableError(
            f"{INSTALL_HINT} (underlying error: {type(error).__name__}: {error})"
        ) from error

    return Coastline, SystemContext, AutoconfFeasibilityChecker


def coastline_available() -> bool:
    """Report whether the COASTLINE recommender engine is importable.

    Returns:
        True if :func:`import_facade` succeeds, False otherwise. Because
        :func:`import_facade` funnels every import-time failure into
        :class:`CoastlineUnavailableError`, this probe is total: a broken engine
        reports False rather than propagating a third-party exception out of what
        reads like a yes/no question.
    """
    try:
        import_facade()
    except CoastlineUnavailableError:
        return False
    return True
