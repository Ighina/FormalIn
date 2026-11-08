"""Paths and environment.

Heavy artefacts (activations, run outputs) live under ``LEANSTEER_SCRATCH``
(default ``/scratch/users/iacopog/leansteer``).  Model weights come from the
Hugging Face cache: a pre-existing ``HF_HOME`` is respected (the shared cache
on this machine already holds all four autoformalizers); otherwise the cache
is placed under scratch.  Import this module before ``transformers``.
"""
from __future__ import annotations

import os
from pathlib import Path

SCRATCH_ROOT = Path(os.environ.get("LEANSTEER_SCRATCH", "/scratch/users/iacopog/leansteer"))
RUNS_DIR = SCRATCH_ROOT / "runs"
FAITHFORM_ROOT = Path(os.environ.get(
    "FAITHFORM_ROOT",
    os.path.expanduser("~/Downloads/45134_FaithformBench_Benchmark_Code And Data Supplement 3"),
))

for _d in (SCRATCH_ROOT, RUNS_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(SCRATCH_ROOT / "hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def run_dir(run_name: str) -> Path:
    d = RUNS_DIR / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d
