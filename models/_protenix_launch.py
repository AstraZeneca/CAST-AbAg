"""Launcher that applies the in-process Protenix hooks, then runs the real runner.

Invoked by ``ProtenixBackend.run`` in place of the real runner, so the hooks attach BEFORE the
model is built, inside each process/rank spawned by ``torch.distributed.run``. It calls
``protenix_hooks.apply_standard_hooks()`` and then ``runpy``s the real runner, forwarding every CLI
arg verbatim.

``apply_standard_hooks`` installs the always-on hooks (sample-name capture, PAE saved as a compact
``*_pae_sample_N.npz``, trunk and per-stage timing to ``stage_wall.json``, per-stage GPU peak to
``stage_mem.json``, chunked confidence) plus the opt-in ones that read their own environment
variables (confidence offload, auto diffusion chunk, OOM skip, FLOP profile, host-RAM trace).

It then enables the two spec-driven hooks when their spec JSON is present::

    PROTENIX_REAL_RUNNER      absolute path to the real runner/inference.py, required
    PROTENIX_LATENT_MOD_SPEC  trunk latent modulation of z_pre
    PROTENIX_DIFFUSION_SPEC   diffusion-side search
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    real_runner = os.environ.get("PROTENIX_REAL_RUNNER")
    if not real_runner:
        sys.stderr.write("PROTENIX_REAL_RUNNER not set; cannot launch Protenix runner.\n")
        return 2

    from models import protenix_hooks
    protenix_hooks.apply_standard_hooks()

    # Forward our argv to the real runner as if it were invoked directly.
    sys.argv = [real_runner, *sys.argv[1:]]
    runpy.run_path(real_runner, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
