#!/usr/bin/env bash
# Run a CPU-only stage (probe, report) as a transient user systemd unit so it outlives the launching shell.
#   scripts/cpu_stage.sh <stage> <config> [extra cli args...]
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=$1; CFG=$2; shift 2
PY=${PY:-/home/iacopog/miniconda3/envs/leansteer/bin/python}
SCRATCH=${LEANSTEER_SCRATCH:-/scratch/users/iacopog/leansteer}
RUN=$(grep -E '^run_name:' "$CFG" | awk '{print $2}')
TAG=$(echo "$*" | tr -c 'a-zA-Z0-9' '_' | cut -c1-40)
LOG=$SCRATCH/logs/${STAGE}_${RUN}${TAG}.log; mkdir -p "$SCRATCH/logs"
UNIT=leansteer-$STAGE-$RUN-$(date +%s)-$RANDOM
systemd-run --user --unit "$UNIT" --quiet -p WorkingDirectory="$PWD" \
  --setenv=PYTHONPATH=. --setenv=OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} --setenv=HF_HUB_OFFLINE=1 \
  --setenv=PATH="$HOME/.elan/bin:$PATH" --setenv=CUDA_VISIBLE_DEVICES= \
  bash -c "$PY -m leansteer.cli $STAGE -c $CFG $* >> '$LOG' 2>&1; echo \"[unit] EXIT \$? at \$(date)\" >> '$LOG'"
echo "unit $UNIT; log $LOG"
