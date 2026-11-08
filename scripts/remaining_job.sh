#!/usr/bin/env bash
# Everything still pending after the 2026-10-27 card loss, sequential and resumable, for one gpuq card slot:
#   1. 300-per-class residual steering (sycophancy for all models; proved for goedel/stepfun/herald)
#   2. steering variants for all models
#   3. DeepSeek-Prover generation + Lean verification for the 300-per-class set and every variant
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/iacopog/miniconda3/envs/leansteer/bin/python
PYV=/home/iacopog/miniconda3/envs/bmuq/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=. PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8
S="--set steer.directions=[dm] steer.alphas=[-8,-4,4,8] steer.random_alphas=[-8,8]"
$PY -m leansteer.cli steer -c configs/default.yaml --models goedel --tasks proved sycophancy $S
$PY -m leansteer.cli steer -c configs/default.yaml --models stepfun --tasks proved sycophancy $S
$PY -m leansteer.cli steer -c configs/default.yaml --models herald --tasks proved sycophancy $S steer.batch_size=16 steer.max_new_tokens=1024
$PY -m leansteer.cli steer -c configs/default.yaml --models kimina --tasks sycophancy $S
for M in kimina goedel stepfun herald; do
  EXTRA=""; [ "$M" = herald ] && EXTRA="steer.batch_size=16 steer.max_new_tokens=1024"
  scripts/variants_job.sh $M $EXTRA
done
GPU_UTIL=0.3
$PYV -m leansteer.cli prove-gen -c configs/default.yaml --set prove.gpu_memory_utilization=$GPU_UTIL
$PY  -m leansteer.cli prove-verify -c configs/default.yaml --set prove.lean_workers=32
for V in ablate cot_only heads_sigma layer_sweep multi_layer; do
  $PYV -m leansteer.cli prove-variant-gen -c configs/default.yaml --variant $V --set prove.gpu_memory_utilization=$GPU_UTIL
  $PY  -m leansteer.cli prove-variant-verify -c configs/default.yaml --variant $V --set prove.lean_workers=32
done
echo "REMAINING PIPELINE DONE $(date)"
