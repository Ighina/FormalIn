"""Stage ``ext-eval``: do the FaithformBench directions generalize?

For each external dataset and model (prompt_last anchor):
  * truth transfer: the existing directions (`perturbed_detect`, `sycophancy`,
    `proved_strict`, `proved`, `proved_unp`) at their selected layers, scored on
    the external true-vs-false label (PRM800K: human step ratings; miniF2F:
    LLM corruption) — no retraining;
  * truth retrained: 5-fold problem-grouped CV probe on the external data at
    the same layer, plus a TF-IDF text baseline;
  * once ``ext-prove-verify`` has run: the same for the verdict tasks defined
    on our own prover's verdicts (proved among true steps; sycophancy =
    true&Proved vs false&Proved; perturbed_detect).
"""
from __future__ import annotations

import json
import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from leansteer.analysis.probes import _Std, _auc, _fit, load_direction
from leansteer.experiment import Run, write_json
from leansteer.run.collect import load_acts

log = logging.getLogger(__name__)
DIRS = ("perturbed_detect", "sycophancy", "proved_strict", "proved", "proved_unp")


def _cv(H, y, g, C, max_iter):
    s = np.zeros(len(y), dtype=np.float32)
    for tr, va in GroupKFold(n_splits=5).split(H, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        std = _Std(H[tr]); s[va] = _fit(std(H[tr]), y[tr], C, max_iter).decision_function(std(H[va]))
    return _auc(y, s)


def _cv_text(texts, y, g):
    s = np.zeros(len(y), dtype=np.float32); texts = np.asarray(texts, dtype=object)
    for tr, va in GroupKFold(n_splits=5).split(texts, y, g):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(texts[tr])
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(vec.transform(texts[tr]), y[tr])
        s[va] = m.decision_function(vec.transform(texts[va]))
    return _auc(y, s)


def run_ext_eval(cfg: dict, name: str) -> dict:
    run = Run(cfg)
    a = cfg["analysis"]; C, max_iter = float(a["probe_C"]), int(a["max_iter"])
    root = run.root / f"ext_{name}"
    steps = pd.read_parquet(root / "steps.parquet")
    vpath = root / "verdicts.json"
    verdicts = json.loads(vpath.read_text()) if vpath.exists() else {}
    out = {"dataset": name, "n": len(steps), "n_true": int(steps.truth.sum()), "models": {}}
    for model in cfg["models"]:
        acts = root / model / "acts"
        if not (acts / "progress.json").exists():
            continue
        index = json.loads((acts / "index.json").read_text()); prog = json.loads((acts / "progress.json").read_text())
        if prog["n_done"] < prog["N"]:
            continue
        X = np.load(acts / "prompt_last.npy", mmap_mode="r"); row_of = {u: i for i, u in enumerate(index["uids"])}
        li_of = {l: i for i, l in enumerate(index["layers"])}
        df = steps[steps.uid.isin(row_of)].reset_index(drop=True)
        idx = np.array([row_of[u] for u in df.uid]); o = np.argsort(idx)
        res: Dict = {"n": len(df), "truth": {}, "verdict_tasks": {}}
        y_truth = df.truth.to_numpy(); g = df.problem_id.to_numpy()
        res["truth"]["tfidf_cv"] = _cv_text(df.step_content.tolist(), y_truth, g)
        for dname in DIRS:
            rp = run.probe_dir(model, "prompt_last", dname) / "result.json"
            if not rp.exists():
                continue
            layer, z = load_direction(run, model, "prompt_last", dname)
            H = np.asarray(X[idx[o], li_of[layer]]).astype(np.float32)[np.argsort(o)]
            res["truth"][f"{dname}_dir_L{layer}"] = _auc(y_truth, H @ z["direction"])
            if dname == "perturbed_detect":
                res["truth"][f"retrained_cv_L{layer}"] = _cv(H, y_truth, g, C, max_iter)
                res["truth"]["layer"] = layer
        # per-layer CV sweep for truth (every 4th layer): where would a probe trained on this data read truth best?
        sweep = {}
        for l in index["layers"][::4] + [index["layers"][-1]]:
            Hl = np.asarray(X[idx[o], li_of[l]]).astype(np.float32)[np.argsort(o)]
            sweep[str(l)] = _cv(Hl, y_truth, g, C, max_iter)
        res["truth"]["cv_by_layer"] = sweep
        best_l = max(sweep, key=lambda k: sweep[k] or 0)
        res["truth"]["best_layer_cv"] = {"layer": int(best_l), "auc": sweep[best_l]}
        # verdict tasks on our prover's verdicts
        if verdicts:
            v = df.uid.map(lambda u: verdicts.get(f"{model}|{u}")).fillna("missing")
            proved = (v == "Proved").to_numpy(); true = y_truth == 1
            tasks = {"proved_unp": (true & proved, true & ~proved & (v != "missing").to_numpy()),
                     "sycophancy": (true & proved, ~true & proved),
                     "proved_strict": (true & proved, ~true & (v == "Refuted").to_numpy()),
                     "failing": ((v.isin(["Proved", "Refuted", "Unknown"])).to_numpy(), (v.isin(["TC_Fail", "AF_Fail"])).to_numpy())}
            res["verdict_rates"] = {"true": v[true].value_counts(normalize=True).round(3).to_dict(), "false": v[~true].value_counts(normalize=True).round(3).to_dict()}
            for task, (pos, neg) in tasks.items():
                m = pos | neg
                if pos.sum() < 10 or neg.sum() < 10:
                    continue
                yy = pos[m].astype(int)
                r = {"n": int(m.sum()), "pos": int(pos.sum())}
                rp = run.probe_dir(model, "prompt_last", task) / "result.json"
                if rp.exists():
                    layer, z = load_direction(run, model, "prompt_last", task)
                    H = np.asarray(X[idx[o], li_of[layer]]).astype(np.float32)[np.argsort(o)]
                    r["bench_dir_auc"] = _auc(yy, H[m] @ z["direction"]); r["retrained_cv_auc"] = _cv(H[m], yy, g[m], C, max_iter); r["layer"] = layer
                r["tfidf_cv"] = _cv_text(df.step_content[m].tolist(), yy, g[m])
                res["verdict_tasks"][task] = r
        out["models"][model] = res
        log.info("[%s/%s] truth: %s", name, model, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["truth"].items()})
        if res["verdict_tasks"]:
            log.info("[%s/%s] verdict tasks: %s", name, model, {t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for t, r in res["verdict_tasks"].items()})
    write_json(root / "eval.json", out)
    return out
