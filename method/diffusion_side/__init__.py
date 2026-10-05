"""Diffusion-side test-time-scaling methods.

One module per family, one section per algorithm. A config names the family in
``method.diffusion_side`` and the algorithm in ``method.diffusion_search.algorithm``::

    best_of_n                 :mod:`best_of_n`              the null lever (N samples x seeds + argmax)
    search                    :mod:`diffusion_search`       noise_search
    guidance                  :mod:`guidance`               fk, tds, cast
    interaction               :mod:`interacting_particles`  af3_red

The families group by MECHANISM: ``search`` explores over candidates or the initial noise,
``guidance`` steers each particle by a per-particle reward, and ``interaction`` couples particles
to earlier structures. The authoritative table is
:data:`~method.diffusion_side.diffusion_search.DIFFUSION_FAMILIES`, imported by both
``run_ablation`` and the job preflight so the runner and the validator cannot drift apart.

:mod:`base_search` holds only what more than one family needs. A primitive used by exactly one
family belongs in that family's module.

All of it is executed by one
:class:`~method.diffusion_side.diffusion_search.DiffusionSearchSpec`, which
``models/protenix_hooks.install_diffusion`` reads to run a coupled search inside the backend's
sampling loop.
"""
from .diffusion_search import (
    ALL_ALGORITHMS,
    DIFFUSION_FAMILIES,
    DiffusionSearchSpec,
    family_of,
)

__all__ = [
    "ALL_ALGORITHMS",
    "DIFFUSION_FAMILIES",
    "DiffusionSearchSpec",
    "family_of",
]
