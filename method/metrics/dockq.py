"""DockQ wrapper.

DockQ (Basu & Wallner, https://github.com/bjornwallner/DockQ) is invoked as a
subprocess. The binary and the model/native paths are always loaded from a
config, never hard-coded, and the raw stdout is persisted alongside the
parsed score.

The DockQ output includes:
    DockQ, Fnat, iRMS, LRMS, native chain mapping.

We return all four numeric fields plus the mapping.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from method import ANTIBODY_CHAINS

DEFAULT_ANTIBODY_CHAINS = ANTIBODY_CHAINS   # the one definition, see method/__init__.py


@dataclass
class DockQResult:
    dockq: Optional[float]           # headline: Total DockQ over all native interfaces
    fnat: Optional[float]
    irms: Optional[float]
    lrms: Optional[float]
    native: str
    model: str
    dockq_abag: Optional[float] = None   # mean DockQ over antibody-antigen interfaces only
    fnat_term: Optional[float] = None    # ab-ag Fnat (unscaled: it IS its own DockQ contribution)
    irms_term: Optional[float] = None    # ab-ag scaled contribution 1/(1+(iRMS/1.5)^2)
    lrms_term: Optional[float] = None    # ab-ag scaled contribution 1/(1+(LRMS/8.5)^2)
    irms_abag: Optional[float] = None    # ab-ag iRMS in Angstroms
    lrms_abag: Optional[float] = None    # ab-ag LRMS in Angstroms
    mapping: Optional[str] = None
    stdout: Optional[str] = None
    error: Optional[str] = None


def _which_dockq(explicit_bin: Optional[str]) -> str:
    """Resolve the DockQ executable. Order: explicit config path → $PATH."""
    if explicit_bin:
        p = Path(explicit_bin).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"dockq binary not found at {p}")
        return str(p)
    found = shutil.which("DockQ") or shutil.which("dockq.py") or shutil.which("dockq")
    if not found:
        raise FileNotFoundError(
            "DockQ not found. Install from https://github.com/bjornwallner/DockQ "
            "and either add it to PATH or set metrics.dockq.bin in the config."
        )
    return found


def _parse_stdout(text: str) -> Dict[str, Optional[float]]:
    """Extract DockQ / Fnat / iRMS / LRMS from DockQ **v2** stdout."""
    def _grab(pattern: str) -> Optional[float]:
        m = re.search(pattern, text, re.M)
        return float(m.group(1)) if m else None

    total = _grab(r"Total DockQ over \d+ native interfaces:\s*([0-9]+\.[0-9]+)")
    return {
        "dockq": total if total is not None else _grab(r"^\s*DockQ:\s*([0-9]+\.[0-9]+)"),
        "fnat": _grab(r"^\s*[Ff]nat:\s*([0-9]+\.[0-9]+)"),
        "irms": _grab(r"^\s*iRMSD?:\s*([0-9]+\.[0-9]+)"),
        "lrms": _grab(r"^\s*LRMSD?:\s*([0-9]+\.[0-9]+)"),
    }


def antigen_mean_dockq(stdout: str, antibody_chains: Sequence[str] = DEFAULT_ANTIBODY_CHAINS) -> Optional[float]:
    """Mean DockQ over the antibody-antigen interfaces in a DockQ v2 stdout."""
    ab = {c.upper() for c in antibody_chains}
    vals: List[float] = []
    for blk in re.split(r"Native chains:", stdout or "")[1:]:
        m = re.search(r"^\s*DockQ:\s*([0-9.]+)", blk, re.M)
        if m and _abag_block_chains(blk, ab):
            vals.append(float(m.group(1)))
    return float(sum(vals) / len(vals)) if vals else None


_CHAIN_RE = re.compile(r"[A-Za-z0-9]+")


def _abag_block_chains(block: str, ab: set):
    """Chain set of one DockQ interface block, or None if the block is NOT antibody-antigen.

    Reads the `Native chains:` line, then the block's `Model chains:` line if the native carries no
    antibody label. Natives are deposited files and label their chains however the depositor did,
    e.g. FoldBench's are A/B, while the MODEL's chains are the ones this pipeline wires and always
    names H/L. Testing only the native silently returns None for every such target.
    """
    for line in block.split("\n")[:2]:
        chains = {c.upper() for c in _CHAIN_RE.findall(line.split(":", 1)[-1])}
        if chains and (chains & ab) and (chains - ab):
            return chains
    return None


# DockQ additive-term scaling (Basu & Wallner 2016): DockQ = (Fnat + s(iRMS,d1) + s(LRMS,d2))/3
# with s(x,d) = 1/(1+(x/d)^2), d1 = 1.5 A (interface RMSD), d2 = 8.5 A (ligand RMSD).
DOCKQ_D1_IRMS = 1.5
DOCKQ_D2_LRMS = 8.5


def antigen_mean_dockq_terms(
    stdout: str, antibody_chains: Sequence[str] = DEFAULT_ANTIBODY_CHAINS,
) -> Optional[dict]:
    """The three additive DockQ terms, each averaged over the antibody-antigen interfaces
        (same interface set as :func:`antigen_mean_dockq`, VH-VL excluded).
    """
    ab = {c.upper() for c in antibody_chains}
    fnat, irms_t, lrms_t = [], [], []
    irms_raw, lrms_raw = [], []
    for blk in re.split(r"Native chains:", stdout or "")[1:]:
        if _abag_block_chains(blk, ab) is None:
            continue                                    # intra-antibody (VH-VL) or antigen-antigen
        mf = re.search(r"^\s*fnat:\s*([0-9.]+)", blk, re.M)
        mi = re.search(r"^\s*iRMSD?:\s*([0-9.]+)", blk, re.M)
        ml = re.search(r"^\s*LRMSD?:\s*([0-9.]+)", blk, re.M)
        if mf and mi and ml:
            fnat.append(float(mf.group(1)))
            irms_t.append(1.0 / (1.0 + (float(mi.group(1)) / DOCKQ_D1_IRMS) ** 2))
            lrms_t.append(1.0 / (1.0 + (float(ml.group(1)) / DOCKQ_D2_LRMS) ** 2))
            irms_raw.append(float(mi.group(1)))
            lrms_raw.append(float(ml.group(1)))
    if not fnat:
        return None
    n = len(fnat)
    return {"fnat_term": sum(fnat) / n, "irms_term": sum(irms_t) / n, "lrms_term": sum(lrms_t) / n,
            "irms_abag": sum(irms_raw) / n, "lrms_abag": sum(lrms_raw) / n}


_ABAG_TERM_KEYS = ("fnat_term", "irms_term", "lrms_term", "irms_abag", "lrms_abag")


def _abag_term_fields(stdout: str, antibody_chains: Sequence[str]) -> dict:
    """The ab-ag term fields for :class:`DockQResult`, Nones when no ab-ag interface exists."""
    t = antigen_mean_dockq_terms(stdout, antibody_chains) or {}
    return {k: t.get(k) for k in _ABAG_TERM_KEYS}


def load_result(path: str | Path,
                antibody_chains: Sequence[str] = DEFAULT_ANTIBODY_CHAINS) -> Optional[DockQResult]:
    """Reconstruct a :class:`DockQResult` from a cached JSON (written by ``dump_result``)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    stdout = d.get("stdout") or ""
    if stdout:
        parsed = _parse_stdout(stdout)
        for k, val in parsed.items():
            if d.get(k) is None:
                d[k] = val
        d["dockq_abag"] = antigen_mean_dockq(stdout, antibody_chains)
        d.update(_abag_term_fields(stdout, antibody_chains))   # backfills every cached json, no re-run
    elif d.get("dockq") is None:
        return None                                       # cached failure → treat as miss, recompute
    fields = DockQResult.__dataclass_fields__
    return DockQResult(**{k: d.get(k) for k in fields})


def compute_dockq(
    model_path: str | Path,
    native_path: str | Path,
    dockq_bin: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout_sec: int = 600,
    antibody_chains: Sequence[str] = DEFAULT_ANTIBODY_CHAINS,
) -> DockQResult:
    """Run DockQ on a single (model, native) pair. Both must be mmCIF or PDB."""
    model_path = Path(model_path)
    native_path = Path(native_path)
    if not model_path.is_file():
        return DockQResult(None, None, None, None, str(native_path), str(model_path),
                           error=f"model not found: {model_path}")
    if not native_path.is_file():
        return DockQResult(None, None, None, None, str(native_path), str(model_path),
                           error=f"native not found: {native_path}")

    bin_ = _which_dockq(dockq_bin)
    cmd = [bin_, str(model_path), str(native_path), *(extra_args or [])]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired as e:
        return DockQResult(None, None, None, None, str(native_path), str(model_path),
                           error=f"timeout after {timeout_sec}s: {e}")

    if proc.returncode != 0:
        return DockQResult(None, None, None, None, str(native_path), str(model_path),
                           stdout=proc.stdout, error=proc.stderr.strip())

    parsed = _parse_stdout(proc.stdout)
    return DockQResult(
        dockq=parsed["dockq"], fnat=parsed["fnat"], irms=parsed["irms"], lrms=parsed["lrms"],
        dockq_abag=antigen_mean_dockq(proc.stdout, antibody_chains),
        native=str(native_path), model=str(model_path), stdout=proc.stdout,
        **_abag_term_fields(proc.stdout, antibody_chains),
    )


def dump_result(result: DockQResult, path: str | Path) -> Path:
    """Persist a single DockQ result as JSON. Overwrites."""
    from results_io import dump_dataclass_json
    return dump_dataclass_json(result, path)
