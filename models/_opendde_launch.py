"""Launcher that applies the in-process hooks to OpenDDE, then runs OpenDDE's own CLI.

The OpenDDE twin of :mod:`models._protenix_launch`. Two differences from it:

1. **It runs inside OpenDDE's OWN virtualenv**, because OpenDDE's dependencies conflict with this
   package's. Put the repo on ``PYTHONPATH`` and invoke this file with the OpenDDE venv python.
   ``models.protenix_hooks`` imports only json/importlib/logging/os/sys at module level so it loads
   there, and ``models/__init__`` swallows the failed ``protenix2`` auto-registration.
2. **There is no runner path to swap.** Protenix is launched by file path, so its launcher replaces
   that path and ``runpy``s the real runner. OpenDDE is launched through a console script calling
   ``runner.cli.opendde_cli()``, so this file calls that entry point directly after hooking. The
   rule is unchanged: patch modules the runner IMPORTS, never classes it re-defines.

``TTS_COFOLD_PKG`` MUST be set before ``models.protenix_hooks`` is imported. The module resolves
the co-folding package name at import time, so setting it afterwards leaves every hook pointed at
``protenix`` and the run proceeds UNSTEERED. It is set at the top of :func:`main` for that reason.

Attach points to re-check on an OpenDDE upgrade, since ``apply_standard_hooks`` is fail-soft per
hook and a moved one leaves the run unsteered without naming a cause: ``opendde.model.opendde``
with ``class OpenDDE`` carrying the five methods the hooks replace,
``opendde.model.modules.{confidence,pairformer}``, ``opendde.model.{sample_confidence,utils}``,
``opendde.data.inference.infer_dataloader``, and ``runner.dumper`` / ``runner.cli`` /
``runner.batch_inference``.
"""
from __future__ import annotations

import os
import sys


def _unshadow_opendde() -> None:
    """Drop THIS file's own directory from ``sys.path`` before anything imports ``opendde``."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p) != here and p not in ("", ".")]


def main() -> int:
    _unshadow_opendde()

    # Must precede the protenix_hooks import -- see the module docstring.
    os.environ["TTS_COFOLD_PKG"] = os.environ.get("TTS_COFOLD_PKG") or "opendde"

    # Prove we got the real package, not our backend module, BEFORE the model is built. Without this
    # the only symptom is a confusing ModuleNotFoundError deep inside a hook.
    import opendde as _odde
    if not hasattr(_odde, "__path__"):
        sys.stderr.write(
            f"_opendde_launch: 'opendde' resolved to {getattr(_odde, '__file__', '?')}, which is a "
            "MODULE, not the OpenDDE package. models/opendde.py is shadowing it -- check sys.path.\n")
        return 2

    from models import protenix_hooks

    if protenix_hooks.cofold_pkg() != "opendde":
        sys.stderr.write(
            f"_opendde_launch: hooks resolved to {protenix_hooks.cofold_pkg()!r}, not 'opendde'. "
            "Refusing to run: the steering hooks would attach to the wrong package (or nothing) and "
            "the run would silently produce an UNSTEERED pool.\n")
        return 2

    # Target-name capture keys every other hook, so it goes first.
    protenix_hooks.patch_sample_name_capture()

    protenix_hooks.apply_standard_hooks()

    # Hand off to OpenDDE's own entry point with our argv (argv[0] dropped, as click expects).
    from runner.cli import opendde_cli

    return int(opendde_cli(args=sys.argv[1:], standalone_mode=False) or 0)


if __name__ == "__main__":
    sys.exit(main())
