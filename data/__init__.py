"""Data layer, model-agnostic.

Targets, dataset loaders, MSA subsample policies. Nothing here knows about
Protenix or OpenDDE, those live in ``models``.
"""
from .target import Chain, Target

__all__ = ["Chain", "Target"]
