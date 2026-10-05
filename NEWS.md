# Release notes

Notable changes per release. CAST-AbAg follows [semantic versioning](https://semver.org/).

## 0.1.0 (unreleased)

First public release. CAST, cross-subsample amortised site tilting, for antibody-antigen
co-folding, with the baselines and the evaluation harness the paper reports.

### Method

- CAST steering over a joint pool. K conditionings share one N-particle filter and the run emits
  `N + K - 1` structures, with the `K - 1` unsteered trunk by-products excluded from every metric
  by `results_io.pool_rows`.
- Site reward over a candidate epitope map, with umbrella stratification, a hinge inside the
  arrival radius and a Huber bound outside it.
- Reference-free hybK12 map construction, pool modes first and then surface tiles ranked by mode
  support, built from an unsteered pool and never from ground truth.
- Baselines. Best-of-N, FK steering, TDS, AF3-ReD, Boltz-Perturb and evolutionary noise search,
  each budget-matched.

### Backends

- Protenix v2 and OpenDDE, selected by name. A per-backend config carries the `_opendde` suffix and
  the job templates resolve it through `TTS_CFG_SUFFIX`.
- Cross-backend peak-memory model with automatic diffusion chunking and confidence CPU-offload.

### Datasets

- Loaders for `cfa50`, `cofoldarena`, `foldbench`, `collected_bad_cases` and `opendde40`.
- The 50 CFA50 reference structures ship in `data/cfa50_targets/`, so `--dataset cfa50` builds from
  the repository alone.
- `foldbench` writes a relabelled ground-truth tree by default, so every dataset presents the same
  H/L/A chain convention to the metrics.

### Evaluation

- DockQ, ipSAE, pDockQ2, LIS and iLIS per sample, with DockQ scored concurrently with sampling.
- `compute_table` reports measured FLOPs per arm over three lenses, trunk, denoiser and confidence,
  and reads steering-unit, pool and handoff-return layouts.

### Known limitations

- Model weights are not included. See the provisioning table in the README.
- There is no unit-test suite. Verification is end to end, and `--dry-run` is the cheap preflight.
