"""Diversity summaries. Fed into the success-against-diversity Pareto plot."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_ca_coords(cif_or_pdb: str | Path) -> np.ndarray:
    """Load Cα coordinates from an mmCIF or PDB file using Biopython."""
    from Bio.PDB import MMCIFParser, PDBParser  # local import to keep import cost low

    path = Path(cif_or_pdb)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    coords: List[np.ndarray] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].coord.copy())
        break  # first model only
    if not coords:
        raise ValueError(f"No Cα atoms found in {path}")
    return np.asarray(coords, dtype=np.float64)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch alignment of P onto Q. Returns (R, t) so that R @ P.T + t ≈ Q.T."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def pairwise_ca_rmsd(paths: List[str | Path]) -> np.ndarray:
    """Return the ``len(paths) × len(paths)`` symmetric Cα-RMSD matrix; NaN on failure."""
    coords: List[Optional[np.ndarray]] = []
    for p in paths:
        try:
            coords.append(_load_ca_coords(p))
        except Exception:
            coords.append(None)

    n = len(paths)
    D = np.full((n, n), np.nan)
    for i in range(n):
        if coords[i] is None:
            continue
        D[i, i] = 0.0
        for j in range(i + 1, n):
            if coords[j] is None:
                continue
            A, B = coords[i], coords[j]
            k = min(len(A), len(B))
            R, t = _kabsch(A[:k], B[:k])
            aligned = (R @ A[:k].T).T + t
            r = float(np.sqrt(np.mean(np.sum((aligned - B[:k]) ** 2, axis=1))))
            D[i, j] = D[j, i] = r
    return D

def diversity_summary(cif_paths: List[str], cluster_threshold_A: float = 2.0) -> Dict[str, Any]:
    """Mean/median pairwise Cα-RMSD and number of clusters at ``cluster_threshold_A`` Å."""
    if len(cif_paths) < 2:
        return {"n": len(cif_paths), "mean_pairwise_rmsd": None,
                "median_pairwise_rmsd": None, "max_pairwise_rmsd": None, "n_clusters": len(cif_paths)}

    D = pairwise_ca_rmsd(cif_paths)
    iu = np.triu_indices_from(D, k=1)
    vals = D[iu]
    vals = vals[~np.isnan(vals)]

    # Union-find on single-linkage edges
    n = len(cif_paths)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(D[i, j]) and D[i, j] < cluster_threshold_A:
                union(i, j)
    n_clusters = len({find(i) for i in range(n)})

    return {
        "n": n,
        "mean_pairwise_rmsd": float(np.mean(vals)) if vals.size else None,
        "median_pairwise_rmsd": float(np.median(vals)) if vals.size else None,
        "max_pairwise_rmsd": float(np.max(vals)) if vals.size else None,
        "n_clusters": int(n_clusters),
        "threshold_A": cluster_threshold_A,
    }
