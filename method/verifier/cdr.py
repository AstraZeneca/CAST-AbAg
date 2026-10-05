"""CDR / paratope-focus signal, is the antibody binding with its CDRs or its framework?

``paratope_cdr_fraction`` is the fraction of the paratope (the antibody residues contacting the
antigen) that lies in a CDR. High is good, low flags a framework-driven interface. Reference:
mlab-boltzopt ``structure_analyzer._compute_paratope_focus_metrics``.

CDR residues are identified by IMGT numbering (ANARCI). The antibody sequence is identical across a
target's diffusion samples, so the CDR residue set is numbered ONCE per target
(``cdr_residues_from_cif``) and reused for every sample's geometry
(``compute_cdr_paratope_4a_heavy``). The per-sample step is pure geometry and needs no ANARCI, so
the per-target map can be precomputed where ANARCI is installed and shipped to a worker that only
does geometry.

TWO PARATOPE CONVENTIONS EXIST IN THIS REPO:

  10 A CA, IMGT token map   used by token_maps.py, screen_paratope.py and verifier/diversity.py,
                            and by THE STEERING CV itself, which resolves ``rmsd_subset: paratope``
                            through $PROTENIX_CDR_MAP.
  4 A heavy atom            used ONLY by ``compute_cdr_paratope_4a_heavy`` below, following
                            mlab-boltzopt rather than this project's convention.

Anything compared against the steered pool wants the 10 A CA quantity. The IMGT half of this module
(``IMGT_CDR_RANGES``, ``_imgt_is_cdr``, ``ANTIBODY_LABELS``, ``cdr_residues_from_cif``) is shared:
token_maps.py imports it to build the 10 A token map.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Set

import numpy as np

from .ipsae import ANTIBODY_LABELS

PARATOPE_CONTACT_A = 4.0        # heavy-atom antibody<->antigen contact cutoff (reference DEFAULT_CONTACT_CUTOFF)
# IMGT CDR position ranges (inclusive): CDR1 27-38, CDR2 56-65, CDR3 105-117.
IMGT_CDR_RANGES = ((27, 38), (56, 65), (105, 117))

_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class CdrSignals:
    # 4 A heavy-atom paratope. The repo-wide convention elsewhere is 10 A CA, see the module
    # docstring, so this field must not be compared against the report or the steered pool.
    paratope_cdr_fraction: Optional[float]   # fraction of paratope residues that are CDR
    n_cdr_contacts: Optional[int]            # paratope residues that are CDR
    n_paratope_residues: Optional[int]       # antibody residues contacting the antigen
    error: Optional[str] = None


def _imgt_is_cdr(pos: int) -> bool:
    return any(lo <= pos <= hi for lo, hi in IMGT_CDR_RANGES)


# Single canonical implementation lives in cif_io, which both this module and its sibling
from method.verifier.cif_io import read_heavy_atoms as _read_heavy


def _chain_sequence(arr, chain: str):
    """Return (seq_one_letter, res_ids) for a chain, in residue order (CA present)."""
    m = arr.chain_id == chain
    rids = arr.res_id[m]
    rnames = arr.res_name[m]
    order, seen = [], set()
    for rid, rn in zip(rids, rnames):
        if rid not in seen:
            seen.add(rid); order.append((int(rid), _AA3TO1.get(rn, "X")))
    order.sort()
    return "".join(a for _, a in order), [r for r, _ in order]


def cdr_residues_from_cif(
    cif_path: str,
    antibody_chains: Optional[Sequence[str]] = None,
    hmmer_path: Optional[str] = None,
) -> Dict[str, Set[int]]:
    """IMGT-number each antibody chain (ANARCI) and return ``{chain: {cdr res_id, ...}}``."""
    arr = _read_heavy(cif_path)
    if arr is None:
        return {}
    try:
        from anarci import anarci
    except ImportError:
        return {}                             # ANARCI not installed, CDR signal unavailable
    chains = list(dict.fromkeys(arr.chain_id.tolist()))
    ab = set(antibody_chains) if antibody_chains else {c for c in chains if c in ANTIBODY_LABELS}
    out: Dict[str, Set[int]] = {}
    kw = {"scheme": "imgt", "output": False}
    if hmmer_path:
        kw["hmmerpath"] = hmmer_path
    for c in chains:
        if c not in ab:
            continue
        seq, res_ids = _chain_sequence(arr, c)
        if len(seq) < 70:                     # too short to be a variable domain
            continue
        try:
            numbered, details, _ = anarci([(c, seq)], **kw)
        except Exception:
            continue
        if not numbered or not numbered[0]:
            continue
        numbering, start, _end = numbered[0][0]
        cdr_ids: Set[int] = set()
        idx = start
        for (pos, _ins), aa in numbering:
            if aa == "-":
                continue
            if 0 <= idx < len(res_ids) and _imgt_is_cdr(pos):
                cdr_ids.add(res_ids[idx])
            idx += 1
        if cdr_ids:
            out[c] = cdr_ids
    return out


def compute_cdr_paratope_4a_heavy(
    cif_path: str,
    antibody_chains: Optional[Sequence[str]],
    cdr_by_chain: Dict[str, Set[int]],
) -> CdrSignals:
    """Per-sample geometry: paratope-CDR fraction at the 4 A HEAVY-ATOM cut, from a precomputed
        per-target CDR map. NOT the 10 A CA convention the steering CV and the report use -- see the
        module docstring before comparing this against either.
    """
    if not cdr_by_chain:
        return CdrSignals(None, None, None, error="no CDR map")
    arr = _read_heavy(cif_path)
    if arr is None:
        return CdrSignals(None, None, None, error="cif unreadable")
    chains = list(dict.fromkeys(arr.chain_id.tolist()))
    ab = set(antibody_chains) if antibody_chains else {c for c in chains if c in ANTIBODY_LABELS}
    ident = bool(ab) and not set(chains).issubset(ab)
    ab_mask = np.isin(arr.chain_id, list(ab)) if ident else np.zeros(arr.array_length(), bool)
    ag_mask = ~ab_mask if ident else np.zeros(arr.array_length(), bool)
    if not ab_mask.any() or not ag_mask.any():
        return CdrSignals(None, None, None, error="no antibody/antigen split")

    from scipy.spatial import cKDTree
    ab_idx = np.where(ab_mask)[0]
    tree_ag = cKDTree(arr.coord[ag_mask])
    # paratope = antibody residues with >=1 heavy atom within the cutoff of the antigen
    paratope: Set = set()
    for i in ab_idx:
        if tree_ag.query_ball_point(arr.coord[i], PARATOPE_CONTACT_A):
            paratope.add((arr.chain_id[i], int(arr.res_id[i])))
    if not paratope:
        return CdrSignals(0.0, 0, 0)          # antibody does not contact the antigen, worst case
    n_cdr = sum(1 for (c, rid) in paratope if rid in cdr_by_chain.get(c, ()))
    frac = n_cdr / len(paratope)
    return CdrSignals(float(frac), int(n_cdr), int(len(paratope)))
