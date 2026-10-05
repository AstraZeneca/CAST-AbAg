"""Structure-quality metrics not provided by the co-folding backend.

- :mod:`dockq`, subprocess wrapper around the DockQ CLI (Basu & Wallner).
- :mod:`site`, RIGHT-SITE metric, did the prediction bind the true epitope patch.
- :mod:`diversity`, pairwise Ca-RMSD summaries and cluster count at a threshold.
- :mod:`score`, the parallel metric stage that drives the above.
"""
from .dockq import compute_dockq
from .diversity import diversity_summary, pairwise_ca_rmsd

__all__ = [
    "compute_dockq",
    "pairwise_ca_rmsd",
    "diversity_summary",
]
