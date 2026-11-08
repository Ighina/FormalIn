"""Stage ``steer``: causal check of the probe directions.

At the task's selected layer ``l`` we add ``alpha * unit * d`` to the residual
stream at the last prompt token and every generated token and regenerate the
formalization greedily (the benchmark's deterministic setting).  ``d`` is a
unit direction, either ``dm`` (difference of the class means on the training
rows — the standard activation-addition direction) or ``probe`` (the
logistic-regression direction, sign fixed so that +alpha points towards the
positive class).  ``unit`` is the distance between the two class means, so
``alpha = 1`` shifts an activation by one class gap.  A random unit direction
with the same magnitudes is the control.  Probe directions are nearly
orthogonal to the mean difference and the class gap *along* them is a few
percent of it, which is why the gap along the probe is not used as the unit.

Without the prover we measure what the generation itself tells us:

    parse_rate      a ``theorem ... sorry`` block was produced (AF_Fail proxy)
    changed         the extracted statement differs from the unsteered one
    sim_base        difflib similarity of the statement to the unsteered one
    typecheck_rate  (optional) Lean accepts the statement (TC_Fail proxy)
    match_true /    sycophancy only: for a *corrupted* step, does the statement
    sim_true        coincide with the model's statement for the *true* step?
                    (coinciding = the error was silently corrected)

Every generation is saved so the statements can later be sent to the prover.
"""
from __future__ import annotations

import difflib
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from leansteer.analysis import lean as leancheck
from leansteer.analysis.probes import load_direction
from leansteer.data.faithform import build_task, extract_statement, load_af, make_uid, problem_split
from leansteer.experiment import Run, write_json
from leansteer.models.backend import HFBackend, Intervention
from leansteer.run.collect import get_backend

log = logging.getLogger(__name__)


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").split())


def _sim(a: Optional[str], b: Optional[str]) -> Optional[float]:
    if not a or not b:
        return None
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _think_split(text: str) -> tuple:
    if "</think>" in text:
        th, rest = text.split("</think>", 1)
        return th, rest
    return "", text


def pick_eval_rows(df: pd.DataFrame, task: str, cfg: dict) -> pd.DataFrame:
    a = cfg["analysis"]; sc = cfg["steer"]
    table = build_task(df, task, int(a["seed"]), balance=task in cfg["balance"])
    assign = problem_split(table.group, a["test_frac"], int(a["seed"]))
    te = table[table.group.map(assign).eq("test")]
    rng = np.random.default_rng(int(a["seed"]))
    parts = []
    base_k = 150   # the selection used by the reported 150-per-class runs; larger selections extend it (nested)
    for lab in (1, 0):
        g = te[te.label == lab]
        k = min(int(sc["n_per_class"]), len(g))
        first = rng.choice(len(g), min(k, base_k, len(g)), replace=False)
        chosen = np.sort(first)
        if k > base_k:   # extras come from a separate generator so the first 150 stay identical
            rest = np.setdiff1d(np.arange(len(g)), first)
            extra = np.random.default_rng(int(a["seed"]) + 1000 + lab).choice(rest, k - len(first), replace=False)
            chosen = np.sort(np.concatenate([first, extra]))
        parts.append(g.iloc[chosen])
    rows = pd.concat(parts).reset_index(drop=True)
    rows["role"] = "eval"
    if task == "sycophancy":
        # true counterparts of the corrupted (negative) rows, generated unsteered as the reference
        neg = rows[rows.label == 0]
        true_uids = {make_uid(p, n, False) for p, n in zip(neg.problem_id, neg.node_id)}
        ref = df[df.uid.isin(true_uids) & ~df.uid.isin(rows.uid)].drop_duplicates("uid").copy()
        ref["label"] = -1; ref["role"] = "reference"; ref["task"] = task; ref["group"] = ref.problem_id
        rows = pd.concat([rows, ref[rows.columns]]).reset_index(drop=True)
    return rows


def _generate_all(backend: HFBackend, prompts: List[str], iv: List[Intervention], bs: int, max_new: int, desc: str) -> List[str]:
    order = np.argsort([-len(p) for p in prompts], kind="stable")
    outs = [None] * len(prompts)
    for i0 in tqdm(range(0, len(order), bs), desc=desc, leave=False):
        rows = order[i0:i0 + bs]
        gen = backend.generate([prompts[i] for i in rows], max_new_tokens=max_new, interventions=iv)
        for i, g in zip(rows, gen):
            outs[i] = g
    return outs


UNIT_KIND = "dm_norm"   # tag stored with every steered generation; rows from another parametrisation are ignored


def class_stats(run: Run, model: str, anchor: str, task: str, layer: int, cfg: dict, df: pd.DataFrame) -> dict:
    """Difference of class means at ``layer`` on the training rows of ``task``."""
    from leansteer.run.collect import load_acts
    a = cfg["analysis"]; seed = int(a["seed"])
    X, row_of, index = load_acts(run, model, anchor)
    li = {l: i for i, l in enumerate(index["layers"])}[layer]
    t = build_task(df, task, seed, balance=task in cfg["balance"])
    t = t[t.uid.isin(row_of)]
    assign = problem_split(t.group, a["test_frac"], seed)
    tr = t[t.group.map(assign).eq("train")]
    idx = np.array([row_of[u] for u in tr.uid]); o = np.argsort(idx)
    H = np.asarray(X[idx[o], li]).astype(np.float32); y = tr.label.to_numpy()[o]
    dm = H[y == 1].mean(0) - H[y == 0].mean(0)
    return {"dm": dm, "dm_norm": float(np.linalg.norm(dm)), "resid_norm": float(np.linalg.norm(H, axis=1).mean()),
            "mean_pos": H[y == 1].mean(0), "mean_neg": H[y == 0].mean(0)}


def head_class_stats(run: Run, model: str, task: str, cfg: dict, df: pd.DataFrame, heads: List[Tuple[int, int]]) -> Dict[Tuple[int, int], dict]:
    """Per selected (layer, head): difference of the class means of that head's
    output on the training rows (prompt_last head store) and its norm."""
    from leansteer.run.collect import load_acts
    a = cfg["analysis"]; seed = int(a["seed"])
    X, row_of, index = load_acts(run, model, "prompt_last", store="acts_heads")
    li = {l: i for i, l in enumerate(index["layers"])}; dh = int(index["head_dim"])
    t = build_task(df, task, seed, balance=task in cfg["balance"])
    t = t[t.uid.isin(row_of)]
    assign = problem_split(t.group, a["test_frac"], seed)
    tr = t[t.group.map(assign).eq("train")]
    idx = np.array([row_of[u] for u in tr.uid]); o = np.argsort(idx)
    y = tr.label.to_numpy()[o]
    out = {}
    for (l, h) in heads:
        H = np.asarray(X[idx[o], li[l], h * dh:(h + 1) * dh]).astype(np.float32)
        dm = H[y == 1].mean(0) - H[y == 0].mean(0)
        out[(l, h)] = {"dm": dm, "dm_norm": float(np.linalg.norm(dm)), "act_norm": float(np.linalg.norm(H, axis=1).mean()),
                       "proj_std": float((H @ (dm / (np.linalg.norm(dm) + 1e-9))).std())}
    return out


def run_steer_heads(cfg: dict, model: str, tasks: Optional[List[str]] = None, backend: Optional[HFBackend] = None) -> dict:
    """ITI-style steering: add ``alpha * |mu+ - mu-|_h * d_h`` to the output of each of
    the top-k heads (ranked by validation AUROC in ``probes_heads``) from the last
    prompt token onward; ``d_h`` is the head's unit diff-of-means (``dm``) or a
    random unit vector (``random``).  Evaluation rows are the same as the
    residual-stream steering and the unsteered generations are reused."""
    run = Run(cfg)
    sc, hc = cfg["steer"], cfg["heads"]
    k = int(hc["steer_k"])
    df = load_af(models=[model])
    backend = backend or get_backend(cfg, model, capture_layers=[])
    rng = np.random.default_rng(int(cfg["analysis"]["seed"]) + 13)
    results = {}
    for task in list(tasks or sc["tasks"]):
        ranking = pd.read_csv(run.probe_dir(model, "prompt_last", task, "probes_heads") / "heads.csv").sort_values("val_auc", ascending=False)
        heads = [(int(r.layer), int(r.head)) for r in ranking.head(k).itertuples()]
        stats = head_class_stats(run, model, task, cfg, df, heads)
        n_heads, dh = backend.head_layout()
        def build(alpha: float, kind: str) -> List[Intervention]:
            per_layer: Dict[int, np.ndarray] = {}
            for (l, h) in heads:
                st = stats[(l, h)]
                if kind == "dm":
                    d = st["dm"] / (st["dm_norm"] + 1e-9)
                else:
                    d = rng.standard_normal(dh).astype(np.float32); d /= np.linalg.norm(d)
                vec = per_layer.setdefault(l, np.zeros(n_heads * dh, dtype=np.float32))
                vec[h * dh:(h + 1) * dh] += alpha * st["dm_norm"] * d
            return [Intervention("head_steer", l, torch.tensor(v), positions=sc["positions"]) for l, v in per_layer.items()]
        rand_seed_state = rng.bit_generator.state
        rows = pick_eval_rows(df, task, cfg)
        if cfg.get("limit"):
            rows = rows.head(int(cfg["limit"]))
        prompts = [backend.prompt(s) for s in rows.step_content]
        log.info("[%s/%s] head steering on %d heads %s; mean |dm| %.3f (mean head act norm %.1f) on %d rows", model, task, k,
                 [f"L{l}H{h}" for l, h in heads[:8]] + (["..."] if k > 8 else []),
                 float(np.mean([st["dm_norm"] for st in stats.values()])), float(np.mean([st["act_norm"] for st in stats.values()])), len(rows))
        d = run.steer_dir(model, task, "steer_heads")
        np.savez(d / f"steer_heads_k{k}.npz", heads=np.array(heads), dm=np.stack([stats[hd]["dm"] for hd in heads]),
                 dm_norm=np.array([stats[hd]["dm_norm"] for hd in heads]), act_norm=np.array([stats[hd]["act_norm"] for hd in heads]))
        gen_path = d / "generations.jsonl"
        # seed with the unsteered generations of the residual experiment (same rows)
        base_path = run.steer_dir(model, task) / "generations.jsonl"
        if not gen_path.exists() and base_path.exists():
            keep = set(rows.uid)
            with open(gen_path, "w") as f:
                for line in base_path.read_text().splitlines():
                    g = json.loads(line)
                    if g["alpha"] == 0.0 and g["uid"] in keep:
                        f.write(line + "\n")
        done = set()
        if gen_path.exists():
            for r in map(json.loads, gen_path.read_text().splitlines()):
                if r.get("uid") and (r["alpha"] == 0.0 or r.get("unit_kind") == UNIT_KIND):
                    done.add((r["direction"], r["alpha"], r["uid"]))
        conds = [("probe", 0.0)] + [("dm", float(al)) for al in hc.get("steer_alphas", sc["alphas"])] + \
                [("random", float(al)) for al in hc.get("steer_random_alphas", sc["random_alphas"])]
        with open(gen_path, "a") as f:
            for dname, alpha in conds:
                is_eval = rows.role.eq("eval").to_numpy() | (alpha == 0.0)
                todo = [i for i in range(len(rows)) if is_eval[i] and (dname, alpha, rows.uid.iloc[i]) not in done]
                if not todo:
                    continue
                rng.bit_generator.state = rand_seed_state   # same random head directions for every alpha
                iv = [] if alpha == 0.0 else build(alpha, "dm" if dname == "dm" else "random")
                outs = _generate_all(backend, [prompts[i] for i in todo], iv, int(sc["batch_size"]),
                                     int(sc["max_new_tokens"]), f"steer-heads {model} {task} {dname} a={alpha:+.0f}")
                for i, g in zip(todo, outs):
                    r = rows.iloc[i]
                    think, rest = _think_split(g)
                    f.write(json.dumps({"task": task, "direction": dname, "alpha": alpha, "layer": None, "heads": [f"L{l}H{h}" for l, h in heads],
                                        "uid": r.uid, "unit_kind": UNIT_KIND, "unit": float(np.mean([st["dm_norm"] for st in stats.values()])),
                                        "label": int(r.label), "role": r.role, "is_perturbed": bool(r.is_perturbed),
                                        "verdict": r.verdict, "step": r.step_content, "generation": g,
                                        "statement": extract_statement(rest), "think_chars": len(think)}) + "\n")
                    f.flush()
        results[task] = score_task(run, model, task, cfg, name="steer_heads")
        write_json(d / "results.json", results[task])
    write_json(run.model_dir(model) / "steer_heads" / "summary.json", results)
    return results


def run_steer(cfg: dict, model: str, tasks: Optional[List[str]] = None, backend: Optional[HFBackend] = None) -> dict:
    run = Run(cfg)
    sc, lc = cfg["steer"], cfg["lean"]
    anchor = sc["anchor"]
    df = load_af(models=[model])
    backend = backend or get_backend(cfg, model, capture_layers=[])
    rng = np.random.default_rng(int(cfg["analysis"]["seed"]) + 7)
    results = {}
    for task in list(tasks or sc["tasks"]):
        layer, z = load_direction(run, model, anchor, task, sc.get("layer"))
        v = z["direction"].astype(np.float32)
        gap = float(z["proj_gap"])
        sign = 1.0 if gap >= 0 else -1.0
        cs = class_stats(run, model, anchor, task, layer, cfg, df)
        unit = cs["dm_norm"]
        dirs = {"dm": cs["dm"] / (unit + 1e-9), "probe": sign * v}
        rand = rng.standard_normal(v.shape).astype(np.float32); rand /= np.linalg.norm(rand)
        dirs["random"] = rand
        rows = pick_eval_rows(df, task, cfg)
        if cfg.get("limit"):
            rows = rows.head(int(cfg["limit"]))
        prompts = [backend.prompt(s) for s in rows.step_content]
        log.info("[%s/%s] steering L%d unit=|dm|=%.3f (gap along probe %.3f, cos(dm,probe) %.2f, resid norm %.1f) on %d rows",
                 model, task, layer, unit, gap, float(dirs["dm"] @ v), cs["resid_norm"], len(rows))
        d = run.steer_dir(model, task)
        np.savez(d / f"steer_directions_L{layer}.npz", layer=layer, dm=cs["dm"], probe=sign * v, random=rand, unit=unit,
                 mean_pos=cs["mean_pos"], mean_neg=cs["mean_neg"], unit_kind=UNIT_KIND)
        gen_path = d / "generations.jsonl"
        done = set()
        if gen_path.exists():
            for r in map(json.loads, gen_path.read_text().splitlines()):
                if r.get("uid") and (r["alpha"] == 0.0 or r.get("unit_kind") == UNIT_KIND):
                    done.add((r["direction"], r["alpha"], r["uid"]))
        conds = [("probe", 0.0)]
        for dname in sc.get("directions", ["dm", "probe"]):
            conds += [(dname, float(al)) for al in sc["alphas"]]
        conds += [("random", float(al)) for al in sc["random_alphas"]]
        with open(gen_path, "a") as f:
            for dname, alpha in conds:
                is_eval = rows.role.eq("eval").to_numpy() | (alpha == 0.0)
                todo = [i for i in range(len(rows)) if is_eval[i] and (dname, alpha, rows.uid.iloc[i]) not in done]
                if not todo:
                    continue
                iv = [] if alpha == 0.0 else [Intervention("steer", layer, torch.tensor(dirs[dname] * alpha * unit),
                                                            positions=sc["positions"])]
                outs = _generate_all(backend, [prompts[i] for i in todo], iv, int(sc["batch_size"]),
                                     int(sc["max_new_tokens"]), f"steer {model} {task} {dname} a={alpha:+.0f}")
                for i, g in zip(todo, outs):
                    r = rows.iloc[i]
                    think, rest = _think_split(g)
                    f.write(json.dumps({"task": task, "direction": dname, "alpha": alpha, "layer": layer, "uid": r.uid,
                                        "unit_kind": UNIT_KIND, "unit": unit,
                                        "label": int(r.label), "role": r.role, "is_perturbed": bool(r.is_perturbed),
                                        "verdict": r.verdict, "step": r.step_content, "generation": g,
                                        "statement": extract_statement(rest), "think_chars": len(think)}) + "\n")
                    f.flush()
        results[task] = score_task(run, model, task, cfg)
        write_json(d / "results.json", results[task])
    write_json(run.model_dir(model) / "steer" / "summary.json", results)
    return results


def current_selection(cfg: dict, model: str, task: str) -> set:
    """uids of the evaluation rows the current config selects (so that scoring
    ignores rows left in the generations file by an earlier ``n_per_class``)."""
    df = load_af(models=[model])
    return set(pick_eval_rows(df, task, cfg).uid)


def score_task(run: Run, model: str, task: str, cfg: dict, name: str = "steer") -> dict:
    d = run.steer_dir(model, task, name)
    gens = [json.loads(l) for l in (d / "generations.jsonl").read_text().splitlines() if l.strip()]
    keep = current_selection(cfg, model, task)
    gens = [g for g in gens if g["uid"] in keep and (g["alpha"] == 0.0 or g.get("unit_kind") == UNIT_KIND)]
    lc = cfg["lean"]
    tc: Dict[int, Optional[bool]] = {}
    if lc.get("enabled") and leancheck.available(lc["project_dir"]):
        cache_p = d / "typecheck.json"
        cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
        todo = [i for i, g in enumerate(gens) if g["statement"] and g["statement"] not in cache]
        res = leancheck.typecheck([gens[i]["statement"] for i in todo], lc["project_dir"], int(lc["timeout"]), int(lc["batch"]))
        for i, ok in zip(todo, res):
            cache[gens[i]["statement"]] = ok
        cache_p.write_text(json.dumps(cache))
        tc = {i: cache.get(g["statement"]) for i, g in enumerate(gens) if g["statement"]}
    base = {g["uid"]: g for g in gens if g["alpha"] == 0.0 and g["direction"] == "probe"}
    true_of = {}
    for g in gens:
        if g["role"] == "reference":
            true_of[g["uid"]] = g
    def true_uid(uid):
        p, n, _ = uid.split("|"); return make_uid(p, n, False)
    groups = defaultdict(list)
    for i, g in enumerate(gens):
        if g["role"] != "eval":
            continue
        b = base.get(g["uid"])
        rec = {"parsed": g["statement"] is not None,
               "changed": (b is not None and _norm(g["statement"]) != _norm(b["statement"])) if g["alpha"] != 0 else False,
               "sim_base": _sim(g["statement"], b["statement"]) if b else None,
               "think_chars": g["think_chars"], "typecheck": tc.get(i)}
        if task == "sycophancy" and g["label"] == 0:
            t = true_of.get(true_uid(g["uid"])) or base.get(true_uid(g["uid"]))
            rec["match_true"] = (t is not None and t["statement"] is not None and _norm(g["statement"]) == _norm(t["statement"]))
            rec["sim_true"] = _sim(g["statement"], t["statement"]) if t else None
        groups[(g["direction"], g["alpha"], g["label"])].append(rec)
    table = []
    for (dname, alpha, label), recs in sorted(groups.items()):
        row = {"direction": dname, "alpha": alpha, "label": label, "n": len(recs),
               "parse_rate": float(np.mean([r["parsed"] for r in recs])),
               "changed_rate": float(np.mean([r["changed"] for r in recs])),
               "sim_base": _mean([r["sim_base"] for r in recs]),
               "think_chars": float(np.mean([r["think_chars"] for r in recs])),
               "typecheck_rate": _mean([r["typecheck"] for r in recs])}
        if task == "sycophancy" and label == 0:
            row["match_true_rate"] = float(np.mean([r["match_true"] for r in recs]))
            row["sim_true"] = _mean([r["sim_true"] for r in recs])
        table.append(row)
    return {"model": model, "task": task, "layer": gens[0].get("layer") if gens else None, "unit_kind": UNIT_KIND,
            "heads": next((g.get("heads") for g in gens if g.get("heads")), None),
            "unit": next((g["unit"] for g in gens if "unit" in g and g["alpha"] != 0.0), None), "table": table}


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None
