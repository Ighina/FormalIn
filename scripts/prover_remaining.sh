#!/usr/bin/env bash
# Prover generation for the 300-per-class steer set and every variant, one shard. Usage: prover_remaining.sh <k/n> <gpu_util>
set -uo pipefail
cd "$(dirname "$0")/.."
SHARD=$1; UTIL=$2
PYV=/home/iacopog/miniconda3/envs/bmuq/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=.
$PYV -m leansteer.cli prove-gen -c configs/default.yaml --set prove.gpu_memory_utilization=$UTIL prove.shard=$SHARD
for V in ablate cot_only heads_sigma layer_sweep multi_layer; do
  $PYV -m leansteer.cli prove-variant-gen -c configs/default.yaml --variant $V --set prove.gpu_memory_utilization=$UTIL prove.shard=$SHARD
done
echo "PROVER SHARD $SHARD DONE $(date)"
