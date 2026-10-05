#!/usr/bin/env bash
# CAST ablation A2-A5 -- entry point for jobs/tts-cast-ablation-pool.yaml.
#
#   jobs/submit_ablation_pool.sh --all --gpus 4 --local --env-file <env.sh>
#   jobs/submit_ablation_pool.sh --a4 --dry-run --local --env-file <env.sh>
#   jobs/submit_ablation_pool.sh --a2 --a3 --gpus 2                 # k8s submit path
#
# ROWS. Each differs from the production CAST arm in ONE field:
#   --a2    guidance_scale=0             is it the TILT?
#   --a3    umbrella_sweep=12            does the umbrella axis earn its cells?
#   --a4    epitope_source=ground_truth  what does the imperfect map cost?    (needs EPI_TRUE)
#   --a5    shuffled map                 does the gain need the CORRECT site? (needs EPI_SHUF)
#   --all   all four
#   --ref   also queue a byte-identical copy of the production CAST arm, so the deltas are inside
#           one submission. Off by default: runs/cast720u4 + runs/cast52 already hold it, and
#           re-running costs a full pool. Pass it when code, seed, checkpoint, map or budget has
#           moved since those ran, or on targets the reference never covered.
#   --ctrl  also queue the stock Best-of-N arm. Off by default: an ablation row is read against the
#           CAST REFERENCE, not against BoN, and the stock control for these targets already exists.
#
# A1 IS NOT A ROW HERE. subsample_pool=1 is byte-identical to the TDS baseline once K=1 forces the
# level correction off, so it is one run reported twice: jobs/submit_baseline_pool.sh --tds.
#
# ALWAYS --dry-run FIRST on a new selection. It exercises the flag parser, the preflight gate, the
# map stamps, the queue seed and one real run_ablation dispatch, and takes no GPU.
#
# This file only picks the template, every option is handled by jobs/submit_steer_pool.sh, and the
# row flags are parsed by the template itself. Run it with --help for the full option list.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
case " $* " in *" --help "*|*" -h "*) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//' ;; esac
TEMPLATE="${TEMPLATE:-$HERE/tts-cast-ablation-pool.yaml}" METHOD="${METHOD:-castabl}" \
  exec "$HERE/submit_steer_pool.sh" "$@"
