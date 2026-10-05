"""MSA subsample policies.

Every subsample is written to disk as an a3m so it can be pointed at by the
backend input JSON (via ``unpairedMsaPath``), so no backend patch is needed.

Policies implemented
--------------------
- ``random``, uniform without replacement
- ``cluster``, greedy Hamming "leader" clustering (~ ``hhfilter -id``): keep a row as a new
                 centroid iff it is > (1-id)*L match-column mismatches from every kept centroid.
                 linclust-in-spirit but self-contained; for the real thing call ``mmseqs linclust``
                 and pass the reps here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class MSASubsampleSpec:
    policy: str            # "random" | "cluster"
    depth: int             # target number of retained rows (including the query row)
    seed: int = 0
    identity_threshold: float = 0.9  # only used by "cluster"

    def label(self) -> str:
        base = f"{self.policy}@d{self.depth}"
        if self.policy == "cluster":
            base += f"_id{self.identity_threshold:.2f}"
        return base + f"_s{self.seed}"


def read_a3m(path: str | Path) -> List[Tuple[str, str]]:
    """Same shape as the upstream Protenix distribution/data_prep/msa/utils.py:read_a3m."""
    rows: List[Tuple[str, str]] = []
    desc: Optional[str] = None
    buf: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if desc is not None:
                    rows.append((desc, "".join(buf)))
                desc, buf = line[1:], []
            else:
                buf.append(line)
    if desc is not None:
        rows.append((desc, "".join(buf)))
    if not rows:
        raise ValueError(f"Empty a3m: {path}")
    return rows


def write_a3m(rows: List[Tuple[str, str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for desc, seq in rows:
            f.write(f">{desc}\n{seq}\n")
    return path


# ---------------------------------------------------------------- policies
def _match_cols(seq: str) -> str:
    """Project an a3m sequence onto its MATCH columns, drop lowercase insertion states,
        keep upper-case residues and '-' gaps. The result is aligned to the query (same length for
        every row), so Hamming can be taken by index.
    """
    return "".join(c for c in seq if not c.islower())


def _hamming_match(a: str, b: str) -> int:
    """Hamming distance over match columns. Inputs MUST be ``_match_cols``-projected (query-aligned,
    equal length); then distance is a simple index comparison."""
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i])


def _random_subsample(rows: List[Tuple[str, str]], depth: int, seed: int) -> List[Tuple[str, str]]:
    if depth >= len(rows):
        return rows
    rng = random.Random(seed)
    tail = rows[1:]
    picked = rng.sample(tail, k=depth - 1)
    return [rows[0], *picked]


def _cluster_centroids(rows: List[Tuple[str, str]], depth: int, seed: int,
                       identity_threshold: float) -> List[Tuple[str, str]]:
    """Greedy clustering: sequentially pick a row and drop everything within
        (1 - identity_threshold) * query_length Hamming distance of it, until
        ``depth`` centroids are found.
    """
    if depth >= len(rows):
        return rows
    rng = random.Random(seed)
    query_match = _match_cols(rows[0][1])
    L = len(query_match) or 1
    cutoff = int((1.0 - identity_threshold) * L)

    tail = list(rows[1:])
    rng.shuffle(tail)  # rotate through in a random order so seed matters
    centroids: List[Tuple[Tuple[str, str], str]] = []   # (original row, match-projected seq)
    for row in tail:
        if len(centroids) >= depth - 1:
            break
        m = _match_cols(row[1])
        if all(_hamming_match(m, cm) > cutoff for _, cm in centroids):
            centroids.append((row, m))
    return [rows[0], *(r for r, _ in centroids)]


_POLICIES = {
    "random": _random_subsample,
    "cluster": _cluster_centroids,
}


def subsample_a3m(a3m_path: str | Path, spec: MSASubsampleSpec, out_path: str | Path) -> Path:
    """Read → subsample → write. The output a3m is what you point Protenix at."""
    rows = read_a3m(a3m_path)
    if spec.policy == "cluster":
        picked = _cluster_centroids(rows, spec.depth, spec.seed, spec.identity_threshold)
    elif spec.policy in _POLICIES:
        picked = _POLICIES[spec.policy](rows, spec.depth, spec.seed)  # type: ignore[operator]
    else:
        raise ValueError(f"Unknown MSA subsample policy: {spec.policy!r}")
    return write_a3m(picked, out_path)
