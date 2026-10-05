"""Full-MSA generation into an MSA database.

The GENERATION half of the MSA story. ``resample.py`` holds the SUBSAMPLING half, and
``method.trunk_side.wire_msa_from_dump`` + ``build_msa_subsample_inputs`` consume the database
downstream.

A separate step is required because the Protenix inference runner does NOT generate MSAs: given an
input with no ``unpairedMsaPath`` it runs at ``N_msa 1``, query only. So MSAs are generated
upstream into a database once, then wired in per experiment.

Engine: the Protenix CLI ``protenix msa``, which reads a Protenix input JSON, searches every
protein chain lacking an MSA, and writes ``{out_dir}/{name}/msa/{chain}/...non_pairing.a3m`` plus
the uniref/bfd components and a paired MSA under ``msa/complex/``. It is CPU and API bound, with no
GPU needed.

Server: point ``MMSEQS_SERVICE_HOST_URL`` at a reachable MMseqs2 server. ``mode="colabfold"`` uses
the public ColabFold server (:data:`DEFAULT_COLABFOLD_URL`), while ``mode="protenix"`` hardwires
its own. ``LAYERNORM_TYPE=torch`` is forced so the CLI import does not trigger a CUDA-JIT compile
on a CPU-only node.
"""
from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_COLABFOLD_URL = "https://api.colabfold.com"


def _has_homolog(path: Path, _cap: int = 2) -> bool:
    """True if an a3m holds the query PLUS at least one homolog."""
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
                if n >= _cap:
                    return True
    return False


def db_has_msa(db_dir: str | Path, target_id: str, require_paired: bool = True) -> bool:
    """True if the DB holds a COMPLETE MSA for ``target_id``. Resumability depends on this."""
    db_dir = Path(db_dir)
    if require_paired:
        msa_dir = db_dir / target_id / "msa"
        pair = msa_dir / "complex" / "pair.a3m"
        if pair.is_file() and _has_homolog(pair):
            return True
        chains = sorted(msa_dir.glob("[0-9]*")) if msa_dir.is_dir() else []
        chain_dirs = [c for c in chains if c.is_dir()]
        if chain_dirs and all((c / "pairing.a3m").is_file() and _has_homolog(c / "pairing.a3m")
                              for c in chain_dirs):
            return True
        if len(chain_dirs) > 1:        # multi-chain and no paired file anywhere -> incomplete
            return False
    for a in db_dir.glob(f"{target_id}/**/non_pairing.a3m"):
        if _has_homolog(a):
            return True
    return False


def msa_input_paths(inputs_dir: str | Path) -> List[Path]:
    """Input JSONs in ``inputs_dir``, EXCLUDING the pipeline's own outputs."""
    return sorted(p for p in Path(inputs_dir).glob("*.json")
                  if not p.stem.endswith("-update-msa"))


def generate_msa(
    input_json: str | Path,
    db_dir: str | Path,
    server_url: str = DEFAULT_COLABFOLD_URL,
    mode: str = "colabfold",
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate the full MSA for ONE Protenix input JSON into the DB."""
    input_json = Path(input_json)
    run_env = dict(os.environ if env is None else env)
    if mode == "colabfold":
        run_env["MMSEQS_SERVICE_HOST_URL"] = server_url
    else:
        run_env.pop("MMSEQS_SERVICE_HOST_URL", None)
    run_env.setdefault("LAYERNORM_TYPE", "torch")   # avoid CUDA-JIT on CPU-only MSA gen
    run_env.setdefault("PROTENIX_LAYERNORM_TYPE", "torch")
    cmd = ["protenix", "msa", "--input", str(input_json),
           "--out_dir", str(db_dir), "--msa_server_mode", mode]
    proc = subprocess.run(cmd, env=run_env, capture_output=True, text=True, timeout=timeout)
    return {
        "target": input_json.stem,
        "returncode": proc.returncode,
        "out_dir": str(db_dir),
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


def generate_msa_db(
    input_paths: Sequence[str | Path],
    db_dir: str | Path,
    server_url: str = DEFAULT_COLABFOLD_URL,
    mode: str = "colabfold",
    skip_existing: bool = True,
    max_workers: int = 1,
    env: Optional[Dict[str, str]] = None,
    shard: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Build/extend an MSA database for many inputs. Resumable + optionally parallel."""
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    input_paths = list(input_paths)
    if shard is not None:
        i, k = shard
        if not (0 <= i < k):
            raise ValueError(f"shard must be (index, count) with 0 <= index < count, got {shard}")
        input_paths = input_paths[i::k]
        logger.info("MSA DB: shard %d/%d -> %d input(s)", i, k, len(input_paths))
    todo: List[Path] = []
    results: List[Dict[str, Any]] = []
    for p in input_paths:
        tid = Path(p).stem
        if skip_existing and db_has_msa(db_dir, tid):
            logger.info("MSA DB: skip %s (already present)", tid)
            results.append({"target": tid, "skipped": True})
        else:
            todo.append(Path(p))

    def _one(p: Path) -> Dict[str, Any]:
        logger.info("MSA DB: generating %s", p.stem)
        r = generate_msa(p, db_dir, server_url=server_url, mode=mode, env=env)
        logger.info("MSA DB: %s → rc=%d", p.stem, r["returncode"])
        return r

    if max_workers <= 1:
        results.extend(_one(p) for p in todo)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_one, p): p for p in todo}
            for fut in as_completed(futs):
                results.append(fut.result())
    return results
