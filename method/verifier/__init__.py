"""Verifier package. Re-exports are LAZY (PEP 562).

Importing a leaf module must not drag the whole scoring stack in. ``method.verifier.cdr`` holds the
IMGT tables that ``experiments/scripts/token_maps.py`` needs to build a CDR map, and that step runs
in an ANARCI environment that has no pandas. Eager re-exports here pulled ``.score`` -> ``results_io``
-> pandas into every such import and made the map builder unrunnable there. The names below still
resolve on attribute access, so ``from method.verifier import score_samples`` is unchanged.
"""
from typing import TYPE_CHECKING

_EXPORTS = {
    "Verifier": ".base",
    "ConfidenceVerifier": ".confidence",
    "IpsaeResult": ".ipsae",
    "compute_ipsae": ".ipsae",
    "soft_ipsae": ".ipsae",
    "IpsaeVerifier": ".ipsae_verifier",
    "CoordSignals": ".interface_scores",
    "InterfaceScores": ".interface_scores",
    "compute_coord_signals": ".interface_scores",
    "compute_interface_scores": ".interface_scores",
    "INTERFACE_SCORES": ".interface_verifier",
    "InterfaceScoreVerifier": ".interface_verifier",
    "consensus_scores": ".consensus",
    "augment_signals": ".score",
    "make_verifier": ".score",
    "score_samples": ".score",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    """Resolve a re-export on first use, importing only the module that defines it."""
    try:
        where = _EXPORTS[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name)) from None
    from importlib import import_module
    return getattr(import_module(where, __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:                      # keeps editors and type checkers resolving the names
    from .base import Verifier
    from .confidence import ConfidenceVerifier
    from .consensus import consensus_scores
    from .interface_scores import (CoordSignals, InterfaceScores, compute_coord_signals,
                                   compute_interface_scores)
    from .interface_verifier import INTERFACE_SCORES, InterfaceScoreVerifier
    from .ipsae import IpsaeResult, compute_ipsae, soft_ipsae
    from .ipsae_verifier import IpsaeVerifier
    from .score import augment_signals, make_verifier, score_samples
