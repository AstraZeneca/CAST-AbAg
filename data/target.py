"""Backend-agnostic target representation.

A :class:`Target` is what dataset loaders produce and what
:class:`models.base.ModelBackend` consumes. It is deliberately schema-poor:
just chains, ground truth (optional), and free-form metadata. Backend-specific
translations (e.g., Protenix's ``proteinChain.unpairedMsaPath``) happen in the
model backend, not here.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

ChainType = Literal["protein", "rna", "dna", "ligand"]


@dataclass
class Chain:
    chain_id: str
    sequence: str                         # AA / nucleotide sequence, or CCD code for ligand
    chain_type: ChainType = "protein"
    # Backend-agnostic MSA source pointer. Backends decide how to consume it.
    msa_a3m_path: Optional[str] = None    # unpaired MSA in a3m format


@dataclass
class Target:
    target_id: str
    chains: List[Chain] = field(default_factory=list)
    ground_truth_path: Optional[str] = None   # CIF or PDB
    task: str = ""                            # free-form category (e.g. "interface_antibody_antigen")
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    def chain_ids(self) -> List[str]:
        return [c.chain_id for c in self.chains]
