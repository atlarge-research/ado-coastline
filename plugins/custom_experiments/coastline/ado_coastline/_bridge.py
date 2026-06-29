# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

"""Locate and import COASTLINE's public recommender facade.

COASTLINE is distributed as ``coastline-recommender`` and ships a stable public
facade -- ``import coastline`` exposes ``coastline.Coastline`` (and
``coastline.recommend``) -- on top of the ``coastline_recommender`` pipeline and the
shared ``coastline_common`` data models. Once it is installed
(``pip install coastline-recommender``, or ``pip install -e <coastline checkout>``
for local dev) a plain ``import coastline`` is all this plugin needs, and that is the
preferred path: the plugin talks to the facade, not to pipeline internals.

For the umbrella monorepo dev layout (``<umbrella>/{ado,coastline,kavier}``) where
COASTLINE may not be pip-installed yet, we fall back to putting a sibling ``coastline``
checkout -- and its ``kavier`` physics engine -- on ``sys.path`` before importing.

The import is performed lazily (inside the experiment functions, not at module
import) so the ado plugin still *registers* even when COASTLINE is absent.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class CoastlineUnavailableError(RuntimeError):
    """Raised when the COASTLINE recommender packages cannot be located/imported."""


def _looks_like_coastline_root(path: Path) -> bool:
    """A directory is a COASTLINE checkout iff it holds the facade, the pipeline
    package, and the shared models package."""
    return (
        (path / "coastline" / "facade.py").is_file()
        and (path / "coastline_recommender").is_dir()
        and (path / "common" / "coastline_common").is_dir()
    )


def find_coastline_root() -> Path | None:
    """Return the COASTLINE repo root, or ``None`` if it cannot be located.

    Resolution order:
      1. ``COASTLINE_ROOT`` environment variable (explicit override).
      2. A sibling ``coastline/`` directory found by walking up from this file
         (the standard umbrella layout: ``<umbrella>/{ado,coastline}``).
    """
    env_root = os.environ.get("COASTLINE_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _looks_like_coastline_root(candidate):
            return candidate
        logger.warning(
            "COASTLINE_ROOT=%s does not look like a COASTLINE checkout "
            "(missing coastline/facade.py, coastline_recommender/, or "
            "common/coastline_common/)",
            env_root,
        )

    here = Path(__file__).resolve()
    for parent in here.parents:
        sibling = parent / "coastline"
        if _looks_like_coastline_root(sibling):
            return sibling
        if _looks_like_coastline_root(parent):
            return parent
    return None


def _inject_sibling_checkout() -> None:
    """Put a sibling COASTLINE checkout (and its kavier engine) on ``sys.path``.

    Idempotent and best-effort: if no sibling checkout is found this is a no-op and
    the subsequent import simply fails with the usual guidance.
    """
    root = find_coastline_root()
    if root is None:
        return

    # coastline + coastline_recommender live at <root>; coastline_common under <root>/common.
    # Append (not insert-at-0): we only need these as a *fallback* when nothing is installed,
    # and the checkout root also holds generically named dirs (api/, models/, config/,
    # examples/, ...) that must not shadow installed packages of the same name.
    for path in (str(root), str(root / "common")):
        if path not in sys.path:
            sys.path.append(path)

    # kavier is coastline's physics engine. When coastline is pip-installed kavier comes
    # in as a dependency; in the umbrella dev layout it is a sibling ``kavier/src`` checkout.
    try:
        import kavier  # noqa: F401
    except Exception:
        env_kavier = os.environ.get("KAVIER_ROOT")
        candidates = []
        if env_kavier:
            candidates.append(Path(env_kavier).expanduser().resolve() / "src")
        candidates.append(root.parent / "kavier" / "src")
        for candidate in candidates:
            if (candidate / "kavier" / "__init__.py").is_file():
                if str(candidate) not in sys.path:
                    sys.path.append(str(candidate))
                break


def _do_import() -> tuple:
    """Import the facade symbols this plugin needs (no path manipulation)."""
    from coastline import Coastline
    from coastline_common.models.context import SystemContext

    # The one non-facade import: coastline's AutoConf feasibility checker, used purely
    # for an *availability probe* (``.available()``) so the plugin can report the backend
    # it actually used and degrade autoconf->rules gracefully. The recommendation itself
    # goes entirely through the public ``coastline.Coastline`` facade.
    from coastline_recommender.predictors.feasibility.autoconf import (
        AutoconfFeasibilityChecker,
    )

    return Coastline, SystemContext, AutoconfFeasibilityChecker


def import_facade() -> tuple:
    """Import COASTLINE's facade and return ``(Coastline, SystemContext,
    AutoconfFeasibilityChecker)``.

    Tries an installed ``coastline`` first (the preferred, published path); on failure
    it injects a sibling umbrella checkout onto ``sys.path`` and retries once. Raises
    :class:`CoastlineUnavailableError` with actionable guidance if both attempts fail.
    """
    try:
        return _do_import()
    except Exception:  # noqa: BLE001 -- retried below, then re-raised as a typed error
        _inject_sibling_checkout()
        try:
            return _do_import()
        except Exception as exc:  # noqa: BLE001
            raise CoastlineUnavailableError(
                "Could not import the COASTLINE recommender. Install it with "
                "'pip install coastline-recommender' (or "
                "'pip install -e <coastline checkout>' for local dev), or place this "
                "plugin in the umbrella layout where 'coastline' is a sibling of 'ado'. "
                "Set COASTLINE_ROOT (and, for the dev layout, KAVIER_ROOT) to override "
                f"the checkout locations. Underlying import error: {exc}"
            ) from exc


def coastline_available() -> bool:
    """True iff the COASTLINE facade can be imported (installed or via the fallback)."""
    try:
        import_facade()
        return True
    except CoastlineUnavailableError:
        return False
