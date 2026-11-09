#!/usr/bin/env bash
# Goedel with the benchmark's <context> prompts (configs/context.yaml, run af_ctx), sequential and resumable,
# sized for a ~19 GB slice of a shared card:
#   collect -> collect-tf -> probe -> probe-tf -> steer (proved, sycophancy; dm ±4/±8 + random ±8, small batches)
# CPU-only follow-ups (controls, uncertainty, abstain, transfer, baselines) run from scripts/context_cpu_job.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/iacopog/miniconda3/envs/leansteer/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=. PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/context.yaml
$PY -m leansteer.cli collect    -c $CFG --set collect.batch_size=${COLLECT_BS:-24}
$PY -m leansteer.cli collect-tf -c $CFG --set tf.batch_size=${TF_BS:-6}
$PY -m leansteer.cli probe      -c $CFG
$PY -m leansteer.cli probe-tf   -c $CFG
touch /scratch/users/iacopog/leansteer/runs/af_ctx/PROBES_DONE
S="--set steer.directions=[dm] steer.alphas=[-8,-4,4,8] steer.random_alphas=[-8,8] steer.batch_size=${STEER_BS:-6}"
$PY -m leansteer.cli steer -c $CFG --tasks proved sycophancy $S
touch /scratch/users/iacopog/leansteer/runs/af_ctx/STEER_DONE
echo "CONTEXT GPU JOB DONE $(date)"
