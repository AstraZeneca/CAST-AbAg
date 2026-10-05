"""Biophysical interface signals: buried surface area, hydrophobic burial, salt bridges, H-bonds.

Antibody-antigen interface descriptors computed from a free dependency (biotite, BSD). They need
HEAVY ATOMS, not just Ca, so they read the cif directly with biotite rather than through the Ca
loader, and are post-hoc signals needing no PAE. Antibody = chains H/L (or ``antibody_chains``),
antigen = the rest.

Signals, all over the antibody-antigen interface, whole chains, no CDR numbering:
  * ``interface_bsa``          buried surface area, ``SASA(ab) + SASA(ag) - SASA(complex)`` in
                               A^2, the standard two-sided interface area (Shrake-Rupley).
  * ``bsa_hydrophobic_frac``   fraction of the BSA contributed by carbon/sulfur atoms, a nonpolar
                               burial proxy.
  * ``n_salt_bridges``         cross-interface Asp/Glu(O) to Lys/Arg/His(N) residue pairs with a
                               charged-atom pair within ``SALT_BRIDGE_A`` (4.0 A).
  * ``n_hbond_contacts``       cross-interface N/O donor-acceptor heavy-atom pairs within
                               ``HBOND_A`` (3.5 A). A protonation-free proxy, since these cifs
                               carry no hydrogens, so there is no angle term.

The BSA is validated against an independent SASA engine (FreeSASA); the salt-bridge and H-bond
counts are geometry validated by hand-computed tests. Reference: mlab-boltzopt
``interface_metrics``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .ipsae import ANTIBODY_LABELS

SASA_PROBE_RADIUS = 1.4
SASA_POINT_NUMBER = 960
SALT_BRIDGE_A = 4.0
HBOND_A = 3.5

# charged side-chain atoms (protonation-independent): anions carry O-, cations carry N+.
_ANION = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
_CATION = {"LYS": {"NZ"}, "ARG": {"NH1", "NH2", "NE"}, "HIS": {"ND1", "NE2"}}


@dataclass
class BiophysSignals:
    interface_bsa: Optional[float]           # Ų, buried surface area over the ab-ag interface
    bsa_hydrophobic_frac: Optional[float]    # fraction of BSA from C/S atoms
    n_salt_bridges: Optional[int]
    n_hbond_contacts: Optional[int]
    error: Optional[str] = None


# Single canonical implementation lives in cif_io, which both this module and its sibling
from method.verifier.cif_io import read_heavy_atoms as _read_heavy


def _sasa(atoms) -> np.ndarray:
    import biotite.structure as struc
    return struc.sasa(atoms, probe_radius=SASA_PROBE_RADIUS, point_number=SASA_POINT_NUMBER)


def compute_biophys(
    cif_path: str,
    antibody_chains: Optional[Sequence[str]] = None,
) -> BiophysSignals:
    """Compute the ab-ag biophysical interface bundle from a cif's heavy atoms."""
    arr = _read_heavy(cif_path)
    if arr is None or arr.array_length() == 0:
        return BiophysSignals(None, None, None, None, error="cif unreadable / no heavy atoms")
    return compute_biophys_array(arr, antibody_chains)


def compute_biophys_array(
    arr,
    antibody_chains: Optional[Sequence[str]] = None,
) -> BiophysSignals:
    """Core: same as :func:`compute_biophys` but on an already-loaded biotite AtomArray
    (heavy atoms). Split out so the geometry can be smoke-tested on synthetic arrays."""
    chains = list(dict.fromkeys(arr.chain_id.tolist()))
    ab = set(antibody_chains) if antibody_chains else {c for c in chains if c in ANTIBODY_LABELS}
    ident = bool(ab) and not set(chains).issubset(ab)
    ab_mask = np.isin(arr.chain_id, list(ab)) if ident else np.zeros(arr.array_length(), bool)
    ag_mask = ~ab_mask if ident else np.zeros(arr.array_length(), bool)
    if not ab_mask.any() or not ag_mask.any():
        return BiophysSignals(None, None, None, None, error="no antibody/antigen split")

    # --- BSA = SASA(ab) + SASA(ag) - SASA(complex), per atom for the hydrophobic split ---
    try:
        s_complex = _sasa(arr)
        s_ab = _sasa(arr[ab_mask])
        s_ag = _sasa(arr[ag_mask])
    except Exception as e:                          # noqa: BLE001
        return BiophysSignals(None, None, None, None, error=f"sasa failed: {e}")
    bsa_ab_atoms = np.nan_to_num(s_ab) - np.nan_to_num(s_complex[ab_mask])
    bsa_ag_atoms = np.nan_to_num(s_ag) - np.nan_to_num(s_complex[ag_mask])
    bsa = float(bsa_ab_atoms.sum() + bsa_ag_atoms.sum())

    hyd_ab = np.isin(arr.element[ab_mask], ["C", "S"])
    hyd_ag = np.isin(arr.element[ag_mask], ["C", "S"])
    bsa_hyd = float(bsa_ab_atoms[hyd_ab].sum() + bsa_ag_atoms[hyd_ag].sum())
    frac = bsa_hyd / bsa if bsa > 1e-6 else None

    # --- salt bridges + H-bond contacts (cross-interface heavy-atom geometry) ---
    from scipy.spatial import cKDTree
    coord = arr.coord
    ab_idx = np.where(ab_mask)[0]
    ag_idx = np.where(ag_mask)[0]

    def _charge_atoms(idx):
        an, cat = [], []
        rn, at = arr.res_name[idx], arr.atom_name[idx]
        rid = arr.res_id[idx]
        for k, i in enumerate(idx):
            if rn[k] in _ANION and at[k] in _ANION[rn[k]]:
                an.append((i, (arr.chain_id[i], rid[k])))
            elif rn[k] in _CATION and at[k] in _CATION[rn[k]]:
                cat.append((i, (arr.chain_id[i], rid[k])))
        return an, cat
    ab_an, ab_cat = _charge_atoms(ab_idx)
    ag_an, ag_cat = _charge_atoms(ag_idx)

    def _bridge_pairs(side_a, side_b):
        pairs = set()
        for ia, ra in side_a:
            for ib, rb in side_b:
                if np.linalg.norm(coord[ia] - coord[ib]) < SALT_BRIDGE_A:
                    pairs.add((ra, rb))
        return pairs
    salt = len(_bridge_pairs(ab_an, ag_cat) | _bridge_pairs(ab_cat, ag_an))

    # H-bond proxy: N/O(ab) within 3.5 Å of N/O(ag), count residue pairs
    no_ab = ab_idx[np.isin(arr.element[ab_idx], ["N", "O"])]
    no_ag = ag_idx[np.isin(arr.element[ag_idx], ["N", "O"])]
    hb = set()
    if len(no_ab) and len(no_ag):
        tree = cKDTree(coord[no_ag])
        for ia in no_ab:
            for jb in tree.query_ball_point(coord[ia], HBOND_A):
                ib = no_ag[jb]
                hb.add(((arr.chain_id[ia], arr.res_id[ia]), (arr.chain_id[ib], arr.res_id[ib])))
    n_hb = len(hb)

    return BiophysSignals(bsa, frac, int(salt), int(n_hb))
