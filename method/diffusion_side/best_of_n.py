"""Best-of-N (diffusion side): N diffusion samples per seed, argmax verifier.

Backend-agnostic and fully wired to the stock backend runner, it is just
``N_sample`` × multiple seeds. Division of labour:

* the N samples are produced by the backend during inference;
* their **verifier scores** are computed by the verifier stage
  (:func:`method.verifier.score_samples`), this module does NOT score;
* **selection**, picking the deployed structure, is Best-of-N's job:
  :func:`select_argmax`, i.e. ``argmax`` over the chosen verifier's scores.

So: verifier = scores, Best-of-N = argmax(scores). The method's parameters (seeds,
samples, steps, cycles) live in ``model.run`` config, not a separate spec.
"""
from __future__ import annotations

from typing import Optional

import logging

import pandas as pd


_log = logging.getLogger(__name__)


def select_argmax(group: pd.DataFrame, rank_by: str = "ranking_score",
                  fallback: Optional[str] = "ranking_score") -> pd.Series:
    """Best-of-N selection: return the argmax-``rank_by`` sample of a per-target group, the deployed structure. Scores are supplied by the verifier stage; this is the
    diffusion-side decision = ``argmax`` over those scores. NaN scores sort last; when
    ``fallback`` is given it breaks the choice for samples whose primary score is missing
    (pass ``fallback=None`` for a pure single-verifier argmax)."""
    if rank_by in group.columns:
        key = group[rank_by]
        if fallback and fallback in group.columns and fallback != rank_by:
            n_filled = int(key.isna().sum())
            if n_filled:
                _log.warning("select_argmax: %d/%d samples have no %r; filling from %r, which is a "
                             "DIFFERENT scale -- those samples compete on the fallback's units.",
                             n_filled, len(key), rank_by, fallback)
            key = key.fillna(group[fallback])
        elif fallback and fallback in group.columns:
            key = key.fillna(group[fallback])
    elif fallback and fallback in group.columns:
        key = group[fallback]
    else:
        # NO VERIFIER AT ALL. Returning row 0 is an arbitrary structure, not a selection, and it
        _log.warning("select_argmax: neither %r nor fallback %r is present; returning the FIRST "
                     "row, which is an arbitrary structure and NOT a Best-of-N selection.",
                     rank_by, fallback)
        return group.iloc[0]
    if key.isna().all():
        _log.warning("select_argmax: %r is present but ALL %d values are NaN; returning the FIRST "
                     "row, which is an arbitrary structure and NOT a Best-of-N selection.",
                     rank_by, len(key))
        return group.iloc[0]
    order = key.sort_values(ascending=False, na_position="last").index
    return group.loc[order[0]] if len(order) else group.iloc[0]
