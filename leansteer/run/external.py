"""Stages for the external datasets (point 6): ``ext-collect`` (prompt-side
anchors), ``ext-generate`` (greedy autoformalization with the benchmark
prompts), then ``ext-prove-gen`` / ``ext-prove-verify`` (prove.py) and
``ext-eval`` (analysis/external_eval.py)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from leansteer.data.external import load_external
from leansteer.data.faithform import extract_statement
from leansteer.experiment import Run
from leansteer.models.backend import HFBackend
from leansteer.run.collect import collect_steps, get_backend
from leansteer.run.steer import _generate_all, _think_split

log = logging.getLogger(__name__)


def ext_dir(run: Run, name: str, model: Optional[str] = None) -> Path:
    d = run.root / f"ext_{name}" / (model or "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_ext_collect(cfg: dict, name: str, model: str, backend: Optional[HFBackend] = None) -> Path:
    run = Run(cfg)
    df = load_external(name, cfg)
    df.drop(columns=["problem"]).to_parquet(ext_dir(run, name) / "steps.parquet")
    return collect_steps(cfg, model, df, ext_dir(run, name, model) / "acts", backend)


def run_ext_generate(cfg: dict, name: str, model: str, backend: Optional[HFBackend] = None) -> Path:
    run = Run(cfg)
    sc = cfg["steer"]
    df = load_external(name, cfg)
    backend = backend or get_backend(cfg, model, capture_layers=[])
    out = ext_dir(run, name, model) / "generations.jsonl"
    done = {json.loads(l)["uid"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = df[~df.uid.isin(done)].reset_index(drop=True)
    log.info("[%s/%s] autoformalizing %d steps (%d already done)", name, model, len(todo), len(done))
    if len(todo):
        prompts = [backend.prompt(s) for s in todo.step_content]
        outs = _generate_all(backend, prompts, [], int(sc["batch_size"]), int(sc["max_new_tokens"]), f"ext {name} {model}")
        with open(out, "a") as f:
            for r, g in zip(todo.itertuples(), outs):
                think, rest = _think_split(g)
                f.write(json.dumps({"uid": r.uid, "problem_id": r.problem_id, "step": r.step_content, "is_perturbed": bool(r.is_perturbed),
                                    "truth": int(r.truth), "label": int(r.truth), "generation": g, "statement": extract_statement(rest),
                                    "think_chars": len(think)}) + "\n")
    return out
