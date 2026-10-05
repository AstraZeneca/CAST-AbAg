"""Model backends, pluggable adapters for different co-folding models.

The rest of the codebase depends only on the :class:`ModelBackend` protocol
in :mod:`models.base`. Registering a new backend is a matter of:

1. Implement :class:`ModelBackend` in a new file under ``models/<name>.py``.
2. Register it via :func:`register_backend` (module-level auto-registration).
3. Reference it from a config with ``model.backend: <name>``.

No trunk-side / diffusion-side method or experiment script should ever import
a concrete backend directly, always go through :func:`get_backend`.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from .base import BackendRunSpec, ModelBackend, PredictionSample

logger = logging.getLogger(__name__)


_REGISTRY: Dict[str, Callable[[Dict[str, Any]], ModelBackend]] = {}


def register_backend(name: str, factory: Callable[[Dict[str, Any]], ModelBackend]) -> None:
    """Associate ``name`` (as used in configs) with a factory that builds the backend."""
    if name in _REGISTRY:
        logger.debug("Backend %r re-registered", name)
    _REGISTRY[name] = factory


def get_backend(name: str, config: Dict[str, Any] | None = None) -> ModelBackend:
    """Instantiate a backend by name. ``config`` is the backend-specific YAML sub-block."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model backend: {name!r}. "
            f"Registered: {sorted(_REGISTRY)}. "
            f"Add one under models/<name>.py and register it."
        )
    return _REGISTRY[name](config or {})


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


# Auto-register the shipped backends. Import failures are non-fatal so an
# environment missing a backend's runtime does not break the others.
def _auto_register() -> None:
    """Import every backend module so its ``register_backend`` call fires."""
    from importlib import import_module
    for mod_name in ("protenix2", "opendde"):
        for candidate in (f"models.{mod_name}", mod_name):
            try:
                import_module(candidate)
                break
            except Exception as e:      # noqa: BLE001, register-time isolation
                logger.debug("Skipping %s auto-registration: %s", candidate, e)


_auto_register()


__all__ = [
    "BackendRunSpec",
    "ModelBackend",
    "PredictionSample",
    "get_backend",
    "list_backends",
    "register_backend",
]
