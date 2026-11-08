"""Stage ``uncertainty``: confidence intervals and split-seed variation for the
headline probe AUROCs.

For each (model, anchor, task) at the validation-selected layer:
  * a problem-clustered bootstrap (resample held-out *problems* with
    replacement, 2000 draws) of the test AUROC of the seed-0 probe;
  * refits over ``analysis.n_seeds`` different problem-level train/test splits
    (same layer), reported as mean +- sd of the test AUROC;
  * the same two quantities for the TF-IDF step-text baseline.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from leansteer.analysis.probes import _Std, _auc, _fit
from leansteer.data.faithform import build_task, load_af, problem_split
from leansteer.experiment import Run, write_json
from leansteer.run.collect import load_acts

log = logging.getLogger(__name__)


def cluster_bootstrap_auc(y: np.ndarray, s: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.std())


def run_uncertainty(cfg: dict, model: str, tasks: Optional[Sequence[str]] = None, anchor: str = "prompt_last") -> dict:
    run = Run(cfg)
    a = cfg["analysis"]; C, max_iter = float(a["probe_C"]), int(a["max_iter"])
    n_seeds = int(a.get("n_seeds", 5))
    df = load_af(models=[model])
    X, row_of, index = load_acts(run, model, anchor)
    li_of = {l: i for i, l in enumerate(index["layers"])}
    out = {}
    for task in list(tasks or cfg["tasks"]):
        rp = run.probe_dir(model, anchor, task) / "result.json"
        if not rp.exists():
            continue
        layer = int(json.loads(rp.read_text())["selected_layer"])
        res = {"layer": layer, "seeds": [], "tfidf_seeds": []}
        for seed in range(n_seeds):
            table = build_task(df, task, seed, balance=task in cfg["balance"])
            table = table[table.uid.isin(row_of)].reset_index(drop=True)
            assign = problem_split(table.group, a["test_frac"], seed)
            te = table.group.map(assign).eq("test").to_numpy()
            tr_t, te_t = table[~te], table[te]
            ytr, yte = tr_t.label.to_numpy(), te_t.label.to_numpy()
            itr = np.array([row_of[u] for u in tr_t.uid]); ite = np.array([row_of[u] for u in te_t.uid])
            Htr = np.asarray(X[np.sort(itr), li_of[layer]]).astype(np.float32)[np.argsort(np.argsort(itr))]
            Hte = np.asarray(X[np.sort(ite), li_of[layer]]).astype(np.float32)[np.argsort(np.argsort(ite))]
            std = _Std(Htr); s_te = _fit(std(Htr), ytr, C, max_iter).decision_function(std(Hte))
            auc = _auc(yte, s_te)
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr_t.step_content)
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(vec.transform(tr_t.step_content), ytr)
            s_tf = m.decision_function(vec.transform(te_t.step_content))
            res["seeds"].append(auc); res["tfidf_seeds"].append(_auc(yte, s_tf))
            if seed == 0:
                lo, hi, sd = cluster_bootstrap_auc(yte, s_te, te_t.group.to_numpy())
                res["bootstrap_ci95"] = [lo, hi]; res["bootstrap_sd"] = sd
                lo, hi, sd = cluster_bootstrap_auc(yte, s_tf, te_t.group.to_numpy())
                res["tfidf_bootstrap_ci95"] = [lo, hi]
                res["n_test"] = int(len(te_t)); res["n_test_problems"] = int(te_t.group.nunique())
        res["mean"] = float(np.mean(res["seeds"])); res["sd"] = float(np.std(res["seeds"], ddof=1)) if n_seeds > 1 else None
        res["tfidf_mean"] = float(np.mean(res["tfidf_seeds"])); res["tfidf_sd"] = float(np.std(res["tfidf_seeds"], ddof=1)) if n_seeds > 1 else None
        out[task] = res
        log.info("[%s/%s] L%d AUROC seed0 %.3f CI95 [%.3f, %.3f] | %d seeds %.3f +- %.3f | tfidf %.3f +- %.3f",
                 model, task, layer, res["seeds"][0], *res["bootstrap_ci95"], n_seeds, res["mean"], res["sd"] or 0, res["tfidf_mean"], res["tfidf_sd"] or 0)
    d = run.model_dir(model) / "uncertainty"; d.mkdir(exist_ok=True)
    write_json(d / "results.json", out)
    return out
