#!/usr/bin/env bash
# Submit a pipeline stage to this host's GPU queue, beside one of your own cards.
#   scripts/gpuq_stage.sh <stage> <config> <card> <GB> [extra cli args...]
# e.g.  scripts/gpuq_stage.sh collect configs/default.yaml 3 30
#       scripts/gpuq_stage.sh steer   configs/default.yaml 3 45 --models goedel
# Output goes to ~/gpuq-logs/<JOB>.log; inside the job the card is cuda:0.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=$1; CFG=$2; CARD=$3; MEM=$4; shift 4
PY=${PY:-/home/iacopog/miniconda3/envs/leansteer/bin/python}
RUN=$(grep -E '^run_name:' "$CFG" | awk '{print $2}')
gpuq submit --devices "$CARD" -m "$MEM" --name "leansteer-$STAGE-$RUN" --detach -- \
  env HF_HUB_OFFLINE=1 PYTHONPATH=. PATH="$HOME/.elan/bin:$PATH" "$PY" -m leansteer.cli "$STAGE" -c "$CFG" "$@"
