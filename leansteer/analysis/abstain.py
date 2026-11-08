"""Stage ``abstain``: a practical use of the pre-generation probe.

The probe cannot rerank samples (it is read before generation), but it can
*abstain*: refuse to formalize the steps whose state says the result will not
be faithful.  On the held-out problems of each model we compute the benchmark
metrics over all step variants

    TPR = P(Proved | true step)      FPR = P(Proved | corrupted step)
    FI  = 2 * TPR * (1 - FPR) / (TPR + (1 - FPR))        (the faithfulness index)

and then drop the q lowest-scoring rows under several scores, reporting the
metrics on the remaining rows, the coverage, and which fraction of the
sycophantic cases (corrupted & Proved) was removed:

    sycophancy_dir   the task's direction (+ = true & proved)
    proved_dir       the ``proved`` direction
    detect_dir       the corruption detector (+ = true step)
    tfidf_detect     a TF-IDF text classifier for corrupted vs true (text baseline)
    random           uniform random scores
"""
from __future__ import annotations

import json
import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from leansteer.analysis.probes import load_direction
from leansteer.data.faithform import PROVED, build_task, load_af, problem_split
from leansteer.experiment import Run, write_json
from leansteer.run.collect import load_acts

log = logging.getLogger(__name__)
QS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def metrics(df: pd.DataFrame) -> dict:
    tr, co = df[~df.is_perturbed], df[df.is_perturbed]
    tpr = float((tr.verdict == PROVED).mean()) if len(tr) else float("nan")
    fpr = float((co.verdict == PROVED).mean()) if len(co) else float("nan")
    fi = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr)) if tpr + (1 - fpr) > 0 else float("nan")
    return {"TPR": tpr, "FPR": fpr, "FI": fi, "n": int(len(df)), "n_true": int(len(tr)), "n_corrupted": int(len(co))}


def run_abstain(cfg: dict, model: str, anchor: str = "prompt_last") -> dict:
    run = Run(cfg)
    a = cfg["analysis"]; seed = int(a["seed"])
    df = load_af(models=[model]).reset_index(drop=True)
    X, row_of, index = load_acts(run, model, anchor)
    li_of = {l: i for i, l in enumerate(index["layers"])}
    df = df[df.uid.isin(row_of)].reset_index(drop=True)
    assign = problem_split(df.problem_id, a["test_frac"], seed)
    te = df[df.problem_id.map(assign).eq("test")].reset_index(drop=True)
    tr = df[df.problem_id.map(assign).eq("train")]
    idx = np.array([row_of[u] for u in te.uid]); o = np.argsort(idx)
    scores: Dict[str, np.ndarray] = {}
    for sname, task in (("sycophancy_dir", "sycophancy"), ("proved_dir", "proved"), ("detect_dir", "perturbed_detect")):
        rp = run.probe_dir(model, anchor, task) / "result.json"
        if not rp.exists():
            continue
        layer, z = load_direction(run, model, anchor, task)
        H = np.asarray(X[idx[o], li_of[layer]]).astype(np.float32)[np.argsort(o)]
        scores[sname] = H @ z["direction"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr.step_content)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(vec.transform(tr.step_content), (~tr.is_perturbed).astype(int))
    scores["tfidf_detect"] = m.decision_function(vec.transform(te.step_content))
    scores["random"] = np.random.default_rng(seed).standard_normal(len(te))
    syc = (te.is_perturbed & (te.verdict == PROVED)).to_numpy()
    out = {"model": model, "base": metrics(te), "n_sycophantic": int(syc.sum()), "curves": {}}
    for sname, s in scores.items():
        curve = []
        order = np.argsort(s)   # lowest scores first = abstain first
        for q in QS:
            k = int(round(q * len(te)))
            keep = np.ones(len(te), bool); keep[order[:k]] = False
            mt = metrics(te[keep])
            mt.update({"q": q, "coverage": float(keep.mean()), "sycophantic_removed": float(syc[~keep].sum() / max(1, syc.sum())),
                       "true_removed": float((~te.is_perturbed.to_numpy())[~keep].mean()) if k else 0.0})
            curve.append(mt)
        out["curves"][sname] = curve
    d = run.model_dir(model) / "abstain"; d.mkdir(exist_ok=True)
    write_json(d / "results.json", out)
    b = out["base"]
    log.info("[%s] base TPR %.3f FPR %.3f FI %.3f | abstain 20%% by sycophancy_dir: FI %.3f (syc removed %.2f) | detect_dir: FI %.3f | tfidf: FI %.3f | random: FI %.3f",
             model, b["TPR"], b["FPR"], b["FI"], out["curves"]["sycophancy_dir"][3]["FI"], out["curves"]["sycophancy_dir"][3]["sycophantic_removed"],
             out["curves"]["detect_dir"][3]["FI"] if "detect_dir" in out["curves"] else -1, out["curves"]["tfidf_detect"][3]["FI"], out["curves"]["random"][3]["FI"])
    return out
