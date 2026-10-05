"""IpsaeVerifier, backend-agnostic sample scorer using ipSAE as the scalar rank.

Depends only on :class:`models.base.ModelBackend`. Any backend that
implements ``iter_predictions`` + ``load_pae_and_coords`` gets ipSAE for free.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from models.base import ModelBackend, PredictionSample

from .ipsae import (
    DEFAULT_DIST_CUTOFF_A,
    DEFAULT_PAE_CUTOFF_A,
    DEFAULT_PLDDT_CUTOFF,
    IpsaeResult,
    compute_ipsae,
    dump_result as dump_ipsae_result,
)


class IpsaeVerifier:
    """Sample selector using ipSAE. Works on any :class:`ModelBackend`."""

    def __init__(
        self,
        backend: ModelBackend,
        predictions_dir: str | Path,
        rank_by: str = "ipsae",
        pae_cutoff: float = DEFAULT_PAE_CUTOFF_A,
        dist_cutoff: float = DEFAULT_DIST_CUTOFF_A,
        plddt_cutoff: float = DEFAULT_PLDDT_CUTOFF,
        fallback_rank_by: str = "ranking_score",
        results_dir: Optional[str | Path] = None,
    ):
        self.backend = backend
        self.predictions_dir = Path(predictions_dir)
        self.rank_by = rank_by
        self.pae_cutoff = pae_cutoff
        self.dist_cutoff = dist_cutoff
        self.plddt_cutoff = plddt_cutoff
        self.fallback_rank_by = fallback_rank_by
        self.results_dir = Path(results_dir) if results_dir else None

    def iter_samples(self, target_id: str) -> List[PredictionSample]:
        """Score every sample of a target with ipSAE (selection is Best-of-N's job)."""
        samples = self.backend.iter_predictions(self.predictions_dir, target_id)
        for s in samples:
            score, err = self._score_one(target_id, s)
            s.ipsae = score
            if err:
                s.extras["ipsae_error"] = err
                s.notes = (s.notes + f"; ipsae_fallback={self.fallback_rank_by}").lstrip("; ")
        return samples

    # ------------------------------------------------------------------ internals
    def _score_one(self, target_id: str, sample: PredictionSample) -> Tuple[Optional[float], Optional[str]]:
        pae, coords, chains, plddt, err = self.backend.load_pae_and_coords(
            self.predictions_dir, target_id, sample.seed, sample.sample_idx,
        )
        if err or pae is None or coords is None or chains is None:
            return None, err or "unknown loader failure"
        result: IpsaeResult = compute_ipsae(
            pae, coords, chains, plddt=plddt,
            pae_cutoff=self.pae_cutoff,
            dist_cutoff=self.dist_cutoff,
            plddt_cutoff=self.plddt_cutoff,
        )
        if self.results_dir is not None:
            tag = f"{target_id}__seed{sample.seed}__s{sample.sample_idx}"
            dump_ipsae_result(result, self.results_dir / "metrics" / "ipsae" / f"{tag}.json")
        return result.ipsae, result.error
