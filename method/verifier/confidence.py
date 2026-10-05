"""Verifier that ranks purely by a native confidence field (e.g. ranking_score).

The backend does all the parsing; we just enumerate and sort. Any of the
normalised fields on :class:`PredictionSample` can be the selector:

    rank_by in {ranking_score, iptm, plddt, ptm, gpde}

For the interface-focused ipSAE verifier, use :class:`IpsaeVerifier` instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from models.base import ModelBackend, PredictionSample


class ConfidenceVerifier:
    """Reads a native confidence field (ranking_score / iptm / ptm / plddt / gpde) as the
    per-sample score. Scoring only, selection is Best-of-N's job
    (:func:`method.diffusion_side.best_of_n.select_argmax`)."""

    def __init__(self, backend: ModelBackend, predictions_dir: str | Path,
                 rank_by: str = "ranking_score"):
        self.backend = backend
        self.predictions_dir = Path(predictions_dir)
        self.rank_by = rank_by

    def iter_samples(self, target_id: str) -> List[PredictionSample]:
        """The backend already parsed the confidence fields onto each sample, just enumerate."""
        return list(self.backend.iter_predictions(self.predictions_dir, target_id))
