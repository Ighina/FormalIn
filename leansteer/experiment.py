"""Run layout and config loading.

runs/<run_name>/<model>/
    acts/<anchor>.npy      [N, n_layers_captured, d] float16 residual stream at the anchor
    acts/index.json        row order (uids), captured layers, hidden size
    acts/meta.jsonl        per-row prompt statistics (one line per stored row)
    acts/progress.json     rows completed (resumable)
    probes/<anchor>/<task>/{layers.csv, result.json, direction_L<k>.npz}
    steer/<task>/{generations.jsonl, results.json}
    report/summary.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from leansteer import config

DEFAULTS: Dict[str, Any] = {
    "run_name": "af_v1",
    "models": ["goedel", "kimina", "herald", "stepfun"],
    "device": "cuda",
    "dtype": "bfloat16",
    "limit": None,
    "collect": {"anchors": ["prompt_last", "step_last", "step_mean"], "batch_size": 64, "layer_stride": 1},
    "tasks": ["proved", "proved_strict", "failing", "sycophancy"],
    "balance": ["proved"],
    "analysis": {"test_frac": 0.3, "val_frac": 0.2, "seed": 0, "probe_C": 0.05, "layer_stride": 2,
                 "anchors": ["prompt_last", "step_mean"], "n_jobs": 8, "max_iter": 500, "n_seeds": 5},
    "steer": {"anchor": "prompt_last", "tasks": ["proved", "sycophancy"], "directions": ["dm", "probe"],
              "alphas": [-8, -4, -2, 2, 4, 8], "random_alphas": [-8, 8], "n_per_class": 48, "max_new_tokens": 2048, "batch_size": 32,
              "positions": "anchor_onward", "layer": None},
    "heads": {"anchors": ["prompt_last", "step_mean"], "batch_size": 64, "layer_stride": 1, "top_k": [1, 4, 16, 64],
              "head_probe_C": 1.0, "steer_k": 16,
              "steer_alphas": [-32, -16, -8, 8, 16, 32], "steer_random_alphas": [-32, 32]},
    "tf": {"anchors": ["output_last", "statement_last", "statement_mean"], "batch_size": 16, "layer_stride": 1,
           "max_output_tokens": 3072, "tasks": ["proved_unp", "failing_unp"]},
    "modernbert": {"epochs": 3, "batch_size": 32, "lr": 3e-5, "max_length": 160, "seeds": 1},
    "prove": {"tasks": ["proved", "sycophancy"],
              "conditions": {"dm": [-8, -4, -2, 2, 4, 8], "probe": [-8, 8], "random": [-8, 8]},
              "n_samples": 2, "max_tokens": 4096, "max_model_len": 8192, "gpu_memory_utilization": 0.3, "gen_batch": 256,
              "lean_workers": 48, "lean_batch": 8},
    "variants": {
        "layer_sweep": {"kind": "resid", "layer_quantiles": [0.25, 0.5, 0.75, 1.0], "alphas": [-8, 8], "random_alphas": [8]},
        "multi_layer": {"kind": "resid_multi", "layer_step": 2, "alphas": [-4, -2, 2, 4], "random_alphas": [-4, 4]},
        "cot_only": {"kind": "resid", "layer_quantiles": [], "positions": "generated", "alphas": [-8, 8], "random_alphas": [-8, 8]},
        "heads_sigma": {"kind": "heads", "k": 16, "unit": "proj_std", "alphas": [-15, 15], "random_alphas": [-15, 15]},
        "ablate": {"kind": "project", "targets": ["mean", "pos_mean", "neg_mean"], "random": True},
    },
    "external": {"prm800k_per_class": 800},
    "relabel": {"prover": "goedel", "per_model": 1500, "n_samples": 1, "max_tokens": 8192, "max_model_len": 12288},
    "lean": {"enabled": False, "project_dir": "/scratch/datasets/lean-stockfish/mathlib",
             "timeout": 900, "batch": 40},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: str | None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) if path else {}
    return _merge(DEFAULTS, cfg or {})


class Run:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = config.run_dir(cfg["run_name"])
        (self.root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def model_dir(self, model: str) -> Path:
        d = self.root / model
        d.mkdir(parents=True, exist_ok=True)
        return d

    def acts_dir(self, model: str) -> Path:
        d = self.model_dir(model) / "acts"
        d.mkdir(exist_ok=True)
        return d

    def probe_dir(self, model: str, anchor: str, task: str, probes_name: str = "probes") -> Path:
        d = self.model_dir(model) / probes_name / anchor / task
        d.mkdir(parents=True, exist_ok=True)
        return d

    def steer_dir(self, model: str, task: str, name: str = "steer") -> Path:
        d = self.model_dir(model) / name / task
        d.mkdir(parents=True, exist_ok=True)
        return d

    def report_dir(self) -> Path:
        d = self.root / "report"
        d.mkdir(exist_ok=True)
        return d


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
