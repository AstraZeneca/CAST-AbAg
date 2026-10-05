"""Verifier protocol.

A verifier's sole job is **scoring**: consume :class:`models.base.PredictionSample`
objects (as produced by the model backend) and attach the post-hoc verifier score.
Selection (argmax over the scores) is the diffusion-side Best-of-N method's job, :func:`method.diffusion_side.best_of_n.select_argmax`.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from models.base import PredictionSample


@runtime_checkable
class Verifier(Protocol):
    rank_by: str

    def iter_samples(self, target_id: str) -> List[PredictionSample]:
        """Enumerate a target's samples with the verifier score attached."""
        ...
