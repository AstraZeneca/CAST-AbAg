#!/usr/bin/env bash
# External baselines -- entry point for jobs/tts-baseline-pool.yaml.
#
#   jobs/submit_baseline_pool.sh --all --gpus 4 --local --env-file <env.sh>
#   jobs/submit_baseline_pool.sh --boltz-perturb --dry-run --local --env-file <env.sh>
#   jobs/submit_baseline_pool.sh --tds --tds-n-seeds=1,2,4 --gpus 4  # the separate-model ladder
#
# ROWS:
#   --fk              FK / SMC (Singhal 2501.06848): reweight+resample, no gradient, its own
#                     6-event window, ONE umbrella at d*=12 A
#   --tds             TDS (Wu 2306.17775) as S SEPARATE models with different MSA seeds, 720
#                     particles EACH. --tds-n-seeds=LIST is a list of MODEL COUNTS, not seed values:
#                     --tds-n-seeds=1,2,4 queues one arm at one model, one at two, one at four
#                     (default 1). --tds-skip-n-seeds=LIST leaves rungs out when extending a ladder
#                     into a NEW run dir. S=1 is the budget-matched arm and doubles as ablation A1.
#                     S>1 costs S x 720 and is a crossover read, never a matched comparison.
#                     Within one --run-dir nothing needs skipping: the queue never re-runs a unit
#                     that is already in done/.
#   --noise-search    Cosyne ES over the initial noise (Jajal 2506.00299), a different family
#   --boltz-perturb   Boltz-Perturb TCP+TBP, always queues its matched control `bpoff` too
#   --bp-tracks       add the single-track attribution arms bptcp / bptbp
#   --all             every row above
#   --ctrl            also queue the stock Best-of-N arm. Off by default: BoN is a different
#                     baseline with its own S x M ladder in jobs/tts-b0b-pool.yaml, and queueing it
#                     here invites crediting the coupled sampler to whichever arm it sits beside.
#
# Best-of-N is NOT here: its S x M ladder is a different shape of sweep and has its own template.
#
# ALWAYS --dry-run FIRST on a new selection. It exercises the flag parser, the preflight gate (which
# is INVERTED for the no-SMC Boltz-Perturb arms), the queue seed and one real run_ablation dispatch
# per distinct config, and takes no GPU.
#
# This file only picks the template, every option is handled by jobs/submit_steer_pool.sh, and the
# row flags are parsed by the template itself. Run it with --help for the full option list.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
case " $* " in *" --help "*|*" -h "*) sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//' ;; esac
TEMPLATE="${TEMPLATE:-$HERE/tts-baseline-pool.yaml}" METHOD="${METHOD:-baseline}" \
  exec "$HERE/submit_steer_pool.sh" "$@"
