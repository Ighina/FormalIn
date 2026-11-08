#!/usr/bin/env bash
# Detached launcher: scripts/launch_stage.sh <stage> <config> <gpu> [extra cli args...]
# Writes logs/<stage>_<run>.log and logs/<stage>_<run>.pid under $LEANSTEER_SCRATCH, records the exit code.
set -uo pipefail
cd "$(dirname "$0")/.."
STAGE=$1; CFG=$2; GPU=$3; shift 3
SCRATCH=${LEANSTEER_SCRATCH:-/scratch/users/iacopog/leansteer}
RUN=$(grep -E '^run_name:' "$CFG" | awk '{print $2}')
LOG=$SCRATCH/logs/${STAGE}_${RUN}.log; PID=$SCRATCH/logs/${STAGE}_${RUN}.pid
mkdir -p "$SCRATCH/logs"
PY=${PY:-/home/iacopog/miniconda3/envs/leansteer/bin/python}
setsid nohup bash -c "
  trap 'echo \"[launcher] got SIGTERM at \$(date)\" >> '$LOG'' TERM
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTHONPATH=. $PY -m leansteer.cli $STAGE -c $CFG $* >> '$LOG' 2>&1
  echo \"[launcher] EXIT \$? at \$(date)\" >> '$LOG'
  rm -f '$PID'
" > /dev/null 2>&1 < /dev/null &
echo $! > "$PID"
echo "launched $STAGE ($RUN) pid $(cat $PID); log $LOG"
