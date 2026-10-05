"""YAML config loading and validation.

Every experiment cell is driven by a single
YAML file so that inputs are never inlined in code. Configs are loaded with
OmegaConf so that:
  - ``!include`` and dotted overrides work
  - relative paths are resolved against the config file's directory

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class RunPaths:
    """Where inputs come from and where results go. All resolved to absolute paths."""

    inputs_dir: Path        # per-design input JSONs (Protenix format)
    predictions_dir: Path   # protenix prediction output (mirrors the upstream Protenix layout)
    results_dir: Path       # aggregated CSV / parquet metrics
    ground_truth_dir: Optional[Path] = None  # optional; needed for DockQ / lDDT


def load_config(path: str | Path, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load a YAML config and merge dotlist overrides."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    OmegaConf.register_new_resolver("env", lambda name, default="": _env(name, default), replace=True)
    OmegaConf.register_new_resolver("cfg_dir", lambda: str(path.parent), replace=True)

    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    # Freeze the origin so downstream code can resolve relative paths.
    OmegaConf.set_struct(cfg, False)
    cfg["_meta"] = {"config_path": str(path), "config_dir": str(path.parent)}
    return cfg


def _env(name: str, default: str) -> str:
    import os
    return os.environ.get(name, default)


def resolve_paths(cfg: DictConfig) -> RunPaths:
    """Turn the ``paths:`` block of a config into an absolute ``RunPaths``."""
    base = Path(cfg["_meta"]["config_dir"])
    p = cfg["paths"]

    def _abs(v: str) -> Path:
        pv = Path(str(v)).expanduser()
        return pv if pv.is_absolute() else (base / pv).resolve()

    return RunPaths(
        inputs_dir=_abs(p["inputs_dir"]),
        predictions_dir=_abs(p["predictions_dir"]),
        results_dir=_abs(p["results_dir"]),
        ground_truth_dir=_abs(p["ground_truth_dir"]) if p.get("ground_truth_dir") else None,
    )


def dump_effective_config(cfg: DictConfig, out_path: str | Path) -> None:
    """Write the fully-resolved config next to the results, every run is reproducible."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), out_path)
    except Exception:                                   # unresolvable interpolation → keep raw
        OmegaConf.save(cfg, out_path)


def get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """OmegaConf.select but with a default that plays nicely with our dict configs."""
    return OmegaConf.select(cfg, dotted, default=default)
