#!/usr/bin/env bash
# Full external-dataset pipeline for one dataset on one card slot. Usage: external_job.sh <prm800k|minif2f> <gpu_util>
set -uo pipefail
cd "$(dirname "$0")/.."
D=$1; UTIL=${2:-0.35}
PY=/home/iacopog/miniconda3/envs/leansteer/bin/python
PYV=/home/iacopog/miniconda3/envs/bmuq/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=. PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8
$PY -m leansteer.cli ext-collect -c configs/default.yaml --ext $D
for M in kimina goedel stepfun herald; do
  EXTRA=""; [ "$M" = herald ] && EXTRA="--set steer.batch_size=16 steer.max_new_tokens=1024"
  $PY -m leansteer.cli ext-generate -c configs/default.yaml --ext $D --models $M $EXTRA
done
$PY -m leansteer.cli ext-eval -c configs/default.yaml --ext $D
$PYV -m leansteer.cli ext-prove-gen -c configs/default.yaml --ext $D --set prove.gpu_memory_utilization=$UTIL
$PY -m leansteer.cli ext-prove-verify -c configs/default.yaml --ext $D --set prove.lean_workers=32
$PY -m leansteer.cli ext-eval -c configs/default.yaml --ext $D
echo "EXTERNAL $D DONE $(date)"
