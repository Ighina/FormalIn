#!/usr/bin/env bash
# Run every steering variant for one model (used inside a gpuq job). Usage: variants_job.sh <model> [extra --set args]
set -uo pipefail
cd "$(dirname "$0")/.."
M=$1; shift
PY=/home/iacopog/miniconda3/envs/leansteer/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=. PATH=$HOME/.elan/bin:$PATH
TASKS="sycophancy"; { [ "$M" = kimina ] || [ "$M" = goedel ]; } && TASKS="sycophancy proved"
for V in ablate cot_only heads_sigma layer_sweep multi_layer; do
  $PY -m leansteer.cli steer-variant -c configs/default.yaml --models $M --variant $V --tasks $TASKS --set steer.n_per_class=150 "$@"
done
