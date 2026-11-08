"""Stage ``probe``: per-layer linear probes from the residual stream to each
FaithformBench verdict task, per autoformalizer and anchor.

For every (model, anchor, task):
  * problems are split train/test (all variants of a problem on one side);
  * at each captured layer a standardised logistic-regression probe is fitted
    on train, the layer is *selected* on a grouped validation split carved out
    of train, and reported on test (AUROC, AP, balanced accuracy);
  * the diff-of-means direction is evaluated alongside;
  * text-only baselines (TF-IDF LR on the step, step length) bound what is
    predictable from the input text without the model;
  * the selected direction is saved for steering, together with the
    projection gap between the two classes (the steering unit).
Cross-task transfer at the selected layer is also reported.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from leansteer.data.faithform import TASKS, build_task, load_af, problem_split
from leansteer.experiment import Run, write_json
from leansteer.run.collect import load_acts

log = logging.getLogger(__name__)


def _auc(y, s) -> Optional[float]:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def _ap(y, s) -> Optional[float]:
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None


def _bacc(y, s, thr=0.0) -> Optional[float]:
    return float(balanced_accuracy_score(y, (s > thr).astype(int))) if len(np.unique(y)) == 2 else None


class _Std:
    def __init__(self, X):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6

    def __call__(self, X):
        return (X - self.mu) / self.sd


def _fit(X, y, C, max_iter):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, tol=1e-4).fit(X, y)


def fit_layer(Xtr, ytr, vtr, Xte, yte, C: float, max_iter: int) -> dict:
    """Xtr/Xte: [n, d] float32.  ``vtr`` marks the validation rows inside train."""
    fitmask = ~vtr
    std = _Std(Xtr[fitmask])
    val_auc = None
    if vtr.any() and len(np.unique(ytr[vtr])) == 2 and len(np.unique(ytr[fitmask])) == 2:
        m = _fit(std(Xtr[fitmask]), ytr[fitmask], C, max_iter)
        val_auc = _auc(ytr[vtr], m.decision_function(std(Xtr[vtr])))
    std = _Std(Xtr)
    clf = _fit(std(Xtr), ytr, C, max_iter)
    w = clf.coef_[0] / std.sd                       # direction in raw activation space
    w_unit = w / (np.linalg.norm(w) + 1e-9)
    s_te = clf.decision_function(std(Xte))
    dm = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    dm_unit = dm / (np.linalg.norm(dm) + 1e-9)
    proj = Xtr @ w_unit
    return {
        "val_auc": val_auc, "test_auc": _auc(yte, s_te), "test_ap": _ap(yte, s_te), "test_bacc": _bacc(yte, s_te),
        "train_auc": _auc(ytr, clf.decision_function(std(Xtr))),
        "dm_test_auc": _auc(yte, Xte @ dm_unit),
        "direction": w_unit.astype(np.float32), "dm_direction": dm_unit.astype(np.float32),
        "intercept": float(clf.intercept_[0]), "mu": std.mu.astype(np.float32), "sd": std.sd.astype(np.float32),
        "coef_std": clf.coef_[0].astype(np.float32),
        "proj_std": float(proj.std()), "proj_gap": float(proj[ytr == 1].mean() - proj[ytr == 0].mean()),
        "proj_pos_mean": float(proj[ytr == 1].mean()), "proj_neg_mean": float(proj[ytr == 0].mean()),
        "resid_norm": float(np.linalg.norm(Xtr, axis=1).mean()),
    }


def text_baselines(tr: pd.DataFrame, te: pd.DataFrame, C: float) -> dict:
    out = {}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr.step_content)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(vec.transform(tr.step_content), tr.label)
    out["tfidf_auc"] = _auc(te.label.to_numpy(), m.decision_function(vec.transform(te.step_content)))
    if "autoformalizer_output" in tr.columns:   # teacher-forced anchors: the fair text baseline also sees the output
        txt = lambda d: (d.step_content + " ||| " + d.autoformalizer_output.fillna("")).tolist()
        vec2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000).fit(txt(tr))
        m2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(vec2.transform(txt(tr)), tr.label)
        out["tfidf_output_auc"] = _auc(te.label.to_numpy(), m2.decision_function(vec2.transform(txt(te))))
    L = lambda d: np.c_[np.log1p(d.step_content.str.len()), np.log1p(d.step_content.str.split().str.len())]
    m = LogisticRegression(class_weight="balanced").fit(L(tr), tr.label)
    out["length_auc"] = _auc(te.label.to_numpy(), m.decision_function(L(te)))
    out["prevalence_test"] = float(te.label.mean())
    return out


def run_probes(cfg: dict, model: str, tasks: Optional[Sequence[str]] = None,
               anchors: Optional[Sequence[str]] = None, store: str = "acts") -> dict:
    """``store`` selects the activation store: ``acts`` (prompt-side anchors) or
    ``acts_tf`` (teacher-forced anchors on unperturbed rows); probe outputs go
    to ``probes/`` or ``probes_tf/`` respectively."""
    run = Run(cfg)
    a = cfg["analysis"]
    C, max_iter, seed = float(a["probe_C"]), int(a["max_iter"]), int(a["seed"])
    df = load_af(models=[model], with_outputs=(store != "acts"))
    tasks = list(tasks or cfg["tasks"])
    anchors = list(anchors or a["anchors"])
    probes_name = "probes" if store == "acts" else "probes_tf"
    summary = {}
    for anchor in anchors:
        X, row_of, index = load_acts(run, model, anchor, store=store)
        layers_all = index["layers"]
        stride = int(a.get("layer_stride", 1))
        use = sorted(set(layers_all[::stride]) | {layers_all[-1]})
        for task in tasks:
            table = build_task(df, task, seed, balance=task in cfg["balance"])
            table = table[table.uid.isin(row_of)].reset_index(drop=True)
            if len(table) < 40 or table.label.sum() < 10 or (table.label == 0).sum() < 10:
                log.warning("[%s/%s/%s] too few rows (%d); skipping", model, anchor, task, len(table))
                continue
            assign = problem_split(table.group, a["test_frac"], seed)
            te = table.group.map(assign).eq("test").to_numpy()
            tr_tab, te_tab = table[~te], table[te]
            val_assign = problem_split(tr_tab.group, a["val_frac"], seed + 1)
            vtr = tr_tab.group.map(val_assign).eq("test").to_numpy()
            ytr, yte = tr_tab.label.to_numpy(), te_tab.label.to_numpy()
            idx_tr = np.array([row_of[u] for u in tr_tab.uid]); idx_te = np.array([row_of[u] for u in te_tab.uid])
            log.info("[%s/%s/%s] n=%d (pos %d) train %d test %d; fitting %d layers", model, anchor, task,
                     len(table), int(table.label.sum()), len(tr_tab), len(te_tab), len(use))
            Xtr_all = np.asarray(X[np.sort(idx_tr)]).astype(np.float32)   # sorted read for memmap efficiency
            Xte_all = np.asarray(X[np.sort(idx_te)]).astype(np.float32)
            # undo the sort
            Xtr_all = Xtr_all[np.argsort(np.argsort(idx_tr))]; Xte_all = Xte_all[np.argsort(np.argsort(idx_te))]
            li_of = {l: i for i, l in enumerate(layers_all)}
            res = Parallel(n_jobs=int(a["n_jobs"]), prefer="threads")(
                delayed(fit_layer)(Xtr_all[:, li_of[l]], ytr, vtr, Xte_all[:, li_of[l]], yte, C, max_iter) for l in use)
            per_layer = {l: r for l, r in zip(use, res)}
            rows = [{"layer": l, **{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}} for l, r in per_layer.items()]
            layers_df = pd.DataFrame(rows)
            sel = max(use, key=lambda l: (per_layer[l]["val_auc"] if per_layer[l]["val_auc"] is not None else -1))
            best = per_layer[sel]
            d = run.probe_dir(model, anchor, task, probes_name)
            layers_df.to_csv(d / "layers.csv", index=False)
            for l in use:   # keep every direction; steering may pick any layer
                r = per_layer[l]
                np.savez(d / f"direction_L{l}.npz", layer=l, direction=r["direction"], dm_direction=r["dm_direction"],
                         mu=r["mu"], sd=r["sd"], coef_std=r["coef_std"], intercept=r["intercept"],
                         proj_std=r["proj_std"], proj_gap=r["proj_gap"], proj_pos_mean=r["proj_pos_mean"],
                         proj_neg_mean=r["proj_neg_mean"], resid_norm=r["resid_norm"])
            result = {
                "model": model, "anchor": anchor, "task": task, "description": TASKS[task].description,
                "n": len(table), "n_pos": int(table.label.sum()), "n_train": len(tr_tab), "n_test": len(te_tab),
                "n_problems_test": int(te_tab.group.nunique()), "selected_layer": sel,
                "selected": {k: v for k, v in best.items() if not isinstance(v, np.ndarray)},
                "best_test_layer": int(layers_df.sort_values("test_auc").layer.iloc[-1]),
                "best_test_auc": float(layers_df.test_auc.max()),
                "baselines": text_baselines(tr_tab, te_tab, C),
                "split": {"test_frac": a["test_frac"], "seed": seed},
            }
            write_json(d / "result.json", result)
            summary[f"{anchor}/{task}"] = result
            log.info("[%s/%s/%s] L%d val %.3f test AUROC %.3f (dm %.3f) | tfidf %.3f length %.3f", model, anchor, task,
                     sel, best["val_auc"] or -1, best["test_auc"] or -1, best["dm_test_auc"] or -1,
                     result["baselines"]["tfidf_auc"] or -1, result["baselines"]["length_auc"] or -1)
        # cross-task transfer at each task's selected layer
        # transfer over every task that has a direction at this anchor (merged with earlier runs)
        have = [t for t in TASKS if (run.probe_dir(model, anchor, t, probes_name) / "result.json").exists()]
        transfer = cross_task_transfer(run, model, anchor, df, cfg, have, X, row_of, index, probes_name)
        write_json(run.model_dir(model) / probes_name / anchor / "transfer.json", transfer)
    sp = run.model_dir(model) / probes_name / "summary.json"
    merged = json.loads(sp.read_text()) if sp.exists() else {}
    merged.update(summary)
    write_json(sp, merged)
    return merged


def load_direction(run: Run, model: str, anchor: str, task: str, layer: Optional[int] = None,
                   probes_name: str = "probes") -> Tuple[int, dict]:
    d = run.probe_dir(model, anchor, task, probes_name)
    if layer is None:
        layer = int(json.loads((d / "result.json").read_text())["selected_layer"])
    z = np.load(d / f"direction_L{layer}.npz")
    return layer, {k: z[k] for k in z.files}


def cross_task_transfer(run, model, anchor, df, cfg, tasks, X, row_of, index, probes_name="probes") -> dict:
    """AUROC of task A's selected direction on task B's *test* rows (B's split)."""
    a = cfg["analysis"]; seed = int(a["seed"])
    li_of = {l: i for i, l in enumerate(index["layers"])}
    out = {}
    for src in tasks:
        layer, z = load_direction(run, model, anchor, src, probes_name=probes_name)
        v = z["direction"]
        for dst in tasks:
            table = build_task(df, dst, seed, balance=dst in cfg["balance"])
            table = table[table.uid.isin(row_of)]
            assign = problem_split(table.group, a["test_frac"], seed)
            te = table[table.group.map(assign).eq("test")]
            idx = np.array([row_of[u] for u in te.uid])
            s = np.asarray(X[np.sort(idx), li_of[layer]]).astype(np.float32)[np.argsort(np.argsort(idx))] @ v
            out[f"{src}->{dst}"] = {"layer": layer, "auc": _auc(te.label.to_numpy(), s)}
    return out
