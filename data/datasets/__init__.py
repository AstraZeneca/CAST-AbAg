"""Dataset loaders. Every loader returns ``List[Target]``."""
from .base import DatasetLoader
from .foldbench import FoldBenchDataset, FoldBenchChain, FoldBenchTarget, dump_target_index
from .collected_bad_cases import (CollectedBadCasesDataset, parse_target_id, parse_seqres)
from .cofoldarena import CFA50Dataset, CoFoldArenaDataset
from .opendde40 import OpenDDE40Dataset

__all__ = [
    "DatasetLoader",
    "FoldBenchDataset",
    "FoldBenchChain",
    "FoldBenchTarget",
    "dump_target_index",
    "CollectedBadCasesDataset",
    "CoFoldArenaDataset",
    "CFA50Dataset",
    "OpenDDE40Dataset",
    "parse_target_id",
    "parse_seqres",
]
