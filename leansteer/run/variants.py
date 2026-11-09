"""Stage ``steer-variant``: steering coverage beyond the single-layer, last-token
residual intervention of ``steer``.

Variants (``cfg["variants"][name]``):

  layer_sweep   residual dm steering at several layers (quantiles of depth plus
                the probe's selected layer), one layer at a time
  multi_layer   residual dm steering at every layer of the upper half at once,
                each layer scaled by its own class-mean distance
  cot_only      residual dm steering at the selected layer applied only to the
                generated tokens (the prefill is untouched)
  heads_sigma   ITI-style head steering with ITI's unit: the standard deviation
                of the head's activations along its direction (alpha ~ 15)
  ablate        projection ablation at the selected layer: the component along
                the dm direction is set to a fixed target (overall train mean,
                positive-class mean or negative-class mean) at the anchor
                onward; a random unit direction is the control

Generations go to ``steer_<variant>/<task>/generations.jsonl`` with the same
record layout as ``steer`` (``direction`` carries the sub-condition, e.g.
``dm_L12``), so ``score_task``, ``prove-gen --which`` and ``dose_response``
apply unchanged.  The unsteered generations of the residual experiment are
reused for the same evaluation rows.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from leansteer.analysis.probes import load_direction
from leansteer.data.faithform import extract_statement, load_af, prompt_inputs
from leansteer.experiment import Run, write_json
from leansteer.models.backend import HFBackend, Intervention
from leansteer.run.collect import get_backend
from leansteer.run.steer import UNIT_KIND, _generate_all, _think_split, class_stats, head_class_stats, pick_eval_rows, score_task

log = logging.getLogger(__name__)


def _seed_from_base(run: Run, model: str, task: str, name: str, rows: pd.DataFrame) -> None:
    gen_path = run.steer_dir(model, task, name) / "generations.jsonl"
    base_path = run.steer_dir(model, task) / "generations.jsonl"
    if not gen_path.exists() and base_path.exists():
        keep = set(rows.uid)
        with open(gen_path, "w") as f:
            for line in base_path.read_text().splitlines():
                g = json.loads(line)
                if g["alpha"] == 0.0 and g["uid"] in keep:
                    f.write(line + "\n")


def _layers_for(spec: dict, n_layers: int, selected: int) -> List[int]:
    ls = set()
    for q in spec.get("layer_quantiles", []):
        ls.add(max(1, int(round(q * n_layers))))
    ls.add(selected)
    return sorted(ls)


def run_steer_variant(cfg: dict, model: str, variant: str, tasks: Optional[List[str]] = None,
                      backend: Optional[HFBackend] = None) -> dict:
    run = Run(cfg)
    sc, spec = cfg["steer"], cfg["variants"][variant]
    name = f"steer_{variant}"
    anchor = sc["anchor"]
    df = load_af(models=[model])
    backend = backend or get_backend(cfg, model, capture_layers=[])
    rng = np.random.default_rng(int(cfg["analysis"]["seed"]) + 17)
    results = {}
    for task in list(tasks or spec.get("tasks") or sc["tasks"]):
        sel_layer, z = load_direction(run, model, anchor, task)
        rows = pick_eval_rows(df, task, cfg)
        if cfg.get("limit"):
            rows = rows.head(int(cfg["limit"]))
        prompts = [backend.prompt(s) for s in prompt_inputs(rows, cfg)]
        d = run.steer_dir(model, task, name)
        _seed_from_base(run, model, task, name, rows)
        gen_path = d / "generations.jsonl"
        done = set()
        for r in map(json.loads, gen_path.read_text().splitlines()):
            if r.get("uid") and (r["alpha"] == 0.0 or r.get("unit_kind") == UNIT_KIND):
                done.add((r["direction"], r["alpha"], r["uid"]))
        conds: List[Tuple[str, float, List[Intervention], dict]] = []   # (direction name, alpha, interventions, extra meta)
        kind = spec["kind"]
        if kind in ("resid", "resid_multi"):
            layers = _layers_for(spec, backend.n_layers, sel_layer) if kind == "resid" else \
                list(range(max(1, backend.n_layers // 2), backend.n_layers + 1, int(spec.get("layer_step", 2))))
            stats = {l: class_stats(run, model, anchor, task, l, cfg, df) for l in layers}
            rand = {l: (lambda v: v / np.linalg.norm(v))(rng.standard_normal(backend.hidden_size).astype(np.float32)) for l in layers}
            positions = spec.get("positions", sc["positions"])
            if kind == "resid":
                for l in layers:
                    unit, dm = stats[l]["dm_norm"], stats[l]["dm"] / (stats[l]["dm_norm"] + 1e-9)
                    for al in spec["alphas"]:
                        conds.append((f"dm_L{l}", float(al), [Intervention("steer", l, torch.tensor(dm * al * unit), positions=positions)], {"layer": l, "unit": unit}))
                    for al in spec.get("random_alphas", []):
                        conds.append((f"random_L{l}", float(al), [Intervention("steer", l, torch.tensor(rand[l] * al * unit), positions=positions)], {"layer": l, "unit": unit}))
            else:
                for al in spec["alphas"]:
                    ivs = [Intervention("steer", l, torch.tensor(stats[l]["dm"] / (stats[l]["dm_norm"] + 1e-9) * al * stats[l]["dm_norm"]), positions=positions) for l in layers]
                    conds.append(("dm_multi", float(al), ivs, {"layers": layers, "unit": float(np.mean([stats[l]["dm_norm"] for l in layers]))}))
                for al in spec.get("random_alphas", []):
                    ivs = [Intervention("steer", l, torch.tensor(rand[l] * al * stats[l]["dm_norm"]), positions=positions) for l in layers]
                    conds.append(("random_multi", float(al), ivs, {"layers": layers, "unit": float(np.mean([stats[l]["dm_norm"] for l in layers]))}))
        elif kind == "heads":
            ranking = pd.read_csv(run.probe_dir(model, "prompt_last", task, "probes_heads") / "heads.csv").sort_values("val_auc", ascending=False)
            heads = [(int(r.layer), int(r.head)) for r in ranking.head(int(spec.get("k", 16))).itertuples()]
            hs = head_class_stats(run, model, task, cfg, df, heads)
            n_heads, dh = backend.head_layout()
            unit_key = spec.get("unit", "proj_std")
            rand_h = {hd: (lambda v: v / np.linalg.norm(v))(rng.standard_normal(dh).astype(np.float32)) for hd in heads}
            def build(al: float, random: bool) -> List[Intervention]:
                per_layer: Dict[int, np.ndarray] = {}
                for (l, h) in heads:
                    st = hs[(l, h)]
                    dvec = rand_h[(l, h)] if random else st["dm"] / (st["dm_norm"] + 1e-9)
                    vec = per_layer.setdefault(l, np.zeros(n_heads * dh, dtype=np.float32))
                    vec[h * dh:(h + 1) * dh] += al * st[unit_key] * dvec
                return [Intervention("head_steer", l, torch.tensor(v), positions=sc["positions"]) for l, v in per_layer.items()]
            meta = {"heads": [f"L{l}H{h}" for l, h in heads], "unit": float(np.mean([hs[hd][unit_key] for hd in heads])), "unit_name": unit_key}
            for al in spec["alphas"]:
                conds.append(("dm_hsig", float(al), build(al, False), meta))
            for al in spec.get("random_alphas", []):
                conds.append(("random_hsig", float(al), build(al, True), meta))
        elif kind == "project":
            st = class_stats(run, model, anchor, task, sel_layer, cfg, df)
            v = (st["dm"] / (st["dm_norm"] + 1e-9)).astype(np.float32)
            X_mu = float(((st["mean_pos"] + st["mean_neg"]) / 2) @ v)
            targets = {"mean": X_mu, "pos_mean": float(st["mean_pos"] @ v), "neg_mean": float(st["mean_neg"] @ v)}
            for tname in spec.get("targets", ["mean"]):
                conds.append((f"project_{tname}", 1.0, [Intervention("project", sel_layer, torch.tensor(v), positions=sc["positions"], target=targets[tname])],
                              {"layer": sel_layer, "target": targets[tname], "unit": st["dm_norm"]}))
            if spec.get("random", True):
                rv = rng.standard_normal(backend.hidden_size).astype(np.float32); rv /= np.linalg.norm(rv)
                # match the data: set the random component to its own train mean (zero-information control)
                conds.append(("project_random", 1.0, [Intervention("project", sel_layer, torch.tensor(rv), positions=sc["positions"],
                                                                     target=float(((st["mean_pos"] + st["mean_neg"]) / 2) @ rv))], {"layer": sel_layer, "unit": st["dm_norm"]}))
        else:
            raise ValueError(kind)
        log.info("[%s/%s/%s] %d conditions on %d rows", model, task, variant, len(conds), len(rows))
        with open(gen_path, "a") as f:
            for dname, alpha, ivs, meta in conds:
                todo = [i for i in range(len(rows)) if rows.role.iloc[i] == "eval" and (dname, alpha, rows.uid.iloc[i]) not in done]
                if not todo:
                    continue
                outs = _generate_all(backend, [prompts[i] for i in todo], ivs, int(sc["batch_size"]), int(sc["max_new_tokens"]),
                                     f"{variant} {model} {task} {dname} a={alpha:+.0f}")
                for i, g in zip(todo, outs):
                    r = rows.iloc[i]
                    think, rest = _think_split(g)
                    f.write(json.dumps({"task": task, "direction": dname, "alpha": alpha, "layer": meta.get("layer"), "variant": variant,
                                        "meta": {k: v for k, v in meta.items() if k != "unit"}, "unit": meta.get("unit"), "uid": r.uid,
                                        "unit_kind": UNIT_KIND, "label": int(r.label), "role": r.role, "is_perturbed": bool(r.is_perturbed),
                                        "verdict": r.verdict, "step": r.step_content, "generation": g,
                                        "statement": extract_statement(rest), "think_chars": len(think)}) + "\n")
                    f.flush()
        results[task] = score_task(run, model, task, cfg, name=name)
        write_json(d / "results.json", results[task])
    write_json(run.model_dir(model) / name / "summary.json", results)
    return results
