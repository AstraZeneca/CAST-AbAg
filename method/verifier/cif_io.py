"""Minimal mmCIF Ca reader shared by the interface + consensus verifiers and the
signal-augmentation path, so the coordinate signals compute from the cif alone (no PAE)
and there is a single canonical parser rather than inlined copies.

Reads Ca coordinates, chain ids (``auth_asym_id`` with ``label_asym_id`` fallback), and
per-residue pLDDT (the B-factor column) by scanning the ``_atom_site`` loop of the first
model. Backend-agnostic: works on any AlphaFold3-style mmCIF (Protenix, OpenDDE, ...).
Mirrors ``models.protenix2._load_ca_and_chains`` but returns ``(None, None, None)`` on any
failure so callers can treat an unreadable sample as simply unscorable.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def load_ca_coords_chains(
    cif_path: str,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray]]:
    """Return ``(coords (N,3), chain_ids [N], plddt (N,))`` for the Cα atoms of the first
    model, or ``(None, None, None)`` if the file is missing/unparseable/empty."""
    cols: List[str] = []
    idx = None
    c_atom = c_chain = c_b = c_x = c_y = c_z = c_model = None
    model0 = None
    coords: List[Tuple[float, float, float]] = []
    chains: List[str] = []
    plddt: List[float] = []
    try:
        with open(cif_path) as fh:
            for line in fh:
                if line.startswith("_atom_site."):
                    cols.append(line.split(".", 1)[1].strip())
                    continue
                if cols and idx is None:                    # first row after the loop header
                    idx = {name: i for i, name in enumerate(cols)}
                    c_atom = idx.get("auth_atom_id", idx.get("label_atom_id"))
                    c_chain = idx.get("auth_asym_id", idx.get("label_asym_id"))
                    c_b = idx.get("B_iso_or_equiv")
                    if c_atom is None or c_chain is None or "Cartn_x" not in idx:
                        return None, None, None
                    c_x, c_y, c_z = idx["Cartn_x"], idx["Cartn_y"], idx["Cartn_z"]
                    c_model = idx.get("pdbx_PDB_model_num")
                if idx is None:
                    continue
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    if coords:
                        break                               # end of the _atom_site loop
                    continue
                p = line.split()
                if len(p) <= c_z or p[c_atom] != "CA":
                    continue
                if c_model is not None:
                    if model0 is None:
                        model0 = p[c_model]
                    elif p[c_model] != model0:
                        break                               # first model only
                coords.append((float(p[c_x]), float(p[c_y]), float(p[c_z])))
                chains.append(p[c_chain])
                plddt.append(float(p[c_b]) if (c_b is not None and c_b < len(p)) else np.nan)
    except (OSError, ValueError, KeyError):
        return None, None, None
    if not coords:
        return None, None, None
    return np.asarray(coords, dtype=np.float64), chains, np.asarray(plddt, dtype=np.float64)


def read_heavy_atoms(cif_path: str):
    """Biotite ``AtomArray`` of the first model's heavy (non-H) atoms, or None if unreadable."""
    try:
        import biotite.structure.io.pdbx as pdbx
        f = pdbx.CIFFile.read(cif_path)
        arr = pdbx.get_structure(f, model=1)
        return arr[arr.element != "H"]
    except Exception:
        return None
