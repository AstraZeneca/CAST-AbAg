# Contributing to CAST-AbAg

Contributions are welcome. This document explains how to report issues and propose changes.

## Reporting issues

Please open a GitHub issue with a clear description of the problem or request, the steps to
reproduce it, and the expected behaviour against what you actually saw. For a bug, include the
package version (`python -c "import importlib.metadata as m; print(m.version('cast-abag'))"`), the
Python version, the platform, and which backend you ran (`protenix2` or `opendde`).

If the problem is in a run rather than in the code, the two most useful artefacts are the worker
log and the unit's `res/config.yaml`, because the written config is what the run actually used and
it often differs from the environment you think you set.

Do **not** include confidential or personal data in issues, pull requests, or fixtures.

## Proposing changes

1. Fork the repository and create a feature branch off the default branch.
2. Install the package with the groups your change touches.
   ```bash
   poetry install                 # core
   poetry install --with guidance # torch, for the differentiable reward path
   poetry install --with cdr      # anarci, for the CDR map
   poetry install --with dev      # pytest and pre-commit
   ```
3. Make the change, keeping it focused.
4. Check it. There is no unit-test suite yet, so verification here is end to end, and the cheapest
   useful check is a dry run, which exercises the checkpoint route, `dataset-check`, the map
   provenance stamps and the per-cell budget without taking a GPU.
   ```bash
   TEMPLATE=jobs/tts-cast-pool.yaml RUN_DIR=/tmp/check \
     bash jobs/submit_steer_pool.sh --gpus 1 --local --dry-run --env-file env.sh
   ```
   For a change that can alter sampling, follow it with a two-target run at a small budget and
   confirm the emitted structure count, `N + K - 1` for a joint pool, and that `dockq` and
   `dockq_abag` are non-null in `per_sample.parquet`.
5. Open a pull request describing what changed and why, and link any related issue. If the change
   can move a reported number, say so explicitly and give the before and after.

## Coding conventions

- Match the surrounding code. Comments explain why something is the way it is, not what the line
  does, and a comment that states a measured fact should say when it was measured.
- Keep a change and its explanation together. A guard that refuses something should say in its own
  message what to do instead.
- Prefer failing loudly over returning an empty result. Several defects in this codebase were
  silent successes, and the guards that now exist are there because of them.
- Public behaviour changes stay backward compatible where possible. Call out any break in the pull
  request description.

## Versioning

CAST-AbAg follows [semantic versioning](https://semver.org/). Notable changes are recorded in
[NEWS.md](NEWS.md).

## Contact

For questions that are not suitable for a public issue, contact a maintainer listed in
[AUTHORS.md](AUTHORS.md).
