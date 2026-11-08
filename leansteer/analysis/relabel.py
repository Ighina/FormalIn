"""Robustness of the probes to the prover: relabel a held-out subset of the
benchmark's *own* formal statements with a second prover (Goedel-Prover-V2)
and compare.

``relabel-select`` picks, per autoformalizer, ``relabel.per_model`` step
variants from the probes' held-out problems (uniformly, all verdict types
except AF_Fail, whose statement does not exist) and reconstructs the formal
statement the benchmark proved: the raw response's theorem (true steps), the
stored statement-with-sorry (TC_Fail / Unknown), the signature of the stored
proof (Proved) or the un-negated negation proof (Refuted).
``relabel-gen`` / ``relabel-verify`` run the second prover on them with the
benchmark protocol; ``relabel-eval`` reports

  * the agreement between the benchmark verdict (DeepSeek-Prover-V2) and the
    new one, per model;
  * for each task, the AUROC of the *existing* probe direction (trained on
    benchmark labels) against the benchmark labels and against the new labels
    on the same rows;
  * cross-validated probes (grouped by problem, selected layer) trained on
    benchmark labels vs. trained on the new labels, on the same rows.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from leansteer.analysis.probes import _Std, _auc, _fit, load_direction
from leansteer.data.faithform import (AF_FAIL, FAILING, PROVED, REFUTED, TASKS, build_task, extract_statement, load_af,
                                      normalise_verdict, problem_split)
from leansteer.experiment import Run, write_json
from leansteer.run.collect import load_acts

log = logging.getLogger(__name__)


def benchmark_statement(row) -> str | None:
    out = row.autoformalizer_output
    if not isinstance(out, str) or not out.strip() or out.strip().startswith("-- Failed"):
        return None
    if row.verdict == REFUTED and row.is_perturbed:
        from leansteer.run.prove import unnegate_full
        return unnegate_full(out)
    return extract_statement(out)


def run_relabel_select(cfg: dict) -> dict:
    from leansteer.run.prove import negate_full, prover_dir, statement_key
    run = Run(cfg)
    rc, a = cfg["relabel"], cfg["analysis"]
    d = prover_dir(run, "relabel", rc.get("prover", "goedel"))
    rng = np.random.default_rng(int(a["seed"]) + 11)
    stmts, rows = {}, []
    for model in cfg["models"]:
        df = load_af(models=[model], with_outputs=True)
        split = problem_split(build_task(df, "proved", int(a["seed"]), balance=False).group, a["test_frac"], int(a["seed"]))
        te = df[df.problem_id.map(split).eq("test") & (df.verdict != AF_FAIL)].copy()
        te["bench_statement"] = [benchmark_statement(r) for r in te.itertuples()]
        n_missing = int(te.bench_statement.isna().sum())
        te = te[te.bench_statement.notna()]
        k = min(int(rc["per_model"]), len(te))
        pick = te.iloc[np.sort(rng.choice(len(te), k, replace=False))]
        log.info("[%s] %d held-out rows with a statement (%d without), sampled %d: %s", model, len(te), n_missing, k,
                 dict(Counter(zip(pick.is_perturbed, pick.verdict))))
        for r in pick.itertuples():
            key = statement_key(r.bench_statement)
            if key not in stmts:
                stmts[key] = {"key": key, "statement": r.bench_statement, "nl": r.step_content, "negation": negate_full(r.bench_statement)}
            rows.append({"model": model, "uid": r.uid, "key": key, "verdict": r.verdict, "is_perturbed": bool(r.is_perturbed)})
    write_json(d / "statements.json", stmts)
    write_json(d / "rows.json", rows)
    log.info("%d rows -> %d unique statements", len(rows), len(stmts))
    return {"n_rows": len(rows), "n_statements": len(stmts)}


def _cv_auc(H, y, groups, C, max_iter, n_splits=5):
    s = np.zeros(len(y), dtype=np.float32)
    for tr, va in GroupKFold(n_splits=n_splits).split(H, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        std = _Std(H[tr]); s[va] = _fit(std(H[tr]), y[tr], C, max_iter).decision_function(std(H[va]))
    return _auc(y, s)


def run_relabel_eval(cfg: dict):
    from leansteer.run.prove import prover_dir
    run = Run(cfg)
    rc, a = cfg["relabel"], cfg["analysis"]
    prover = rc.get("prover", "goedel")
    d = prover_dir(run, "relabel", prover)
    rows = json.loads((d / "rows.json").read_text())
    verified = json.loads((d / "verified.json").read_text())
    C, max_iter, seed = float(a["probe_C"]), int(a["max_iter"]), int(a["seed"])
    anchor = "prompt_last"
    out = {"prover": prover, "models": {}}
    for model in cfg["models"]:
        mrows = [r for r in rows if r["model"] == model and r["key"] in verified]
        if not mrows:
            continue
        df = load_af(models=[model]).set_index("uid")
        new = pd.DataFrame(mrows).set_index("uid")
        new["new_verdict"] = [verified[k]["verdict"] for k in new.key]
        agree = pd.crosstab(new.verdict, new.new_verdict)
        res = {"n": len(new), "agreement": float((new.verdict == new.new_verdict).mean()),
               "proved_agreement": float(((new.verdict == PROVED) == (new.new_verdict == PROVED)).mean()),
               "confusion": {str(k): {str(c): int(v) for c, v in agree.loc[k].items()} for k in agree.index}, "tasks": {}}
        X, row_of, index = load_acts(run, model, anchor)
        li_of = {l: i for i, l in enumerate(index["layers"])}
        sub = df.loc[new.index].copy()
        for task in cfg["tasks"]:
            t = TASKS[task]
            def labels(frame):
                pos = t.positive(frame).to_numpy(); neg = t.negative(frame).to_numpy() & ~pos
                return pos, neg
            pos_b, neg_b = labels(sub)
            alt = sub.copy(); alt["verdict"] = new.new_verdict.reindex(sub.index).to_numpy()
            pos_n, neg_n = labels(alt)
            keep_b, keep_n = pos_b | neg_b, pos_n | neg_n
            layer, z = load_direction(run, model, anchor, task)
            v = z["direction"]
            idx = np.array([row_of[u] for u in sub.index]); o = np.argsort(idx)
            H = np.asarray(X[idx[o], li_of[layer]]).astype(np.float32)[np.argsort(o)]
            s = H @ v
            g = sub.problem_id.to_numpy()
            r = {"layer": layer, "n_bench": int(keep_b.sum()), "pos_bench": int(pos_b.sum()), "n_new": int(keep_n.sum()), "pos_new": int(pos_n.sum())}
            if keep_b.sum() >= 40 and pos_b.sum() >= 10 and neg_b.sum() >= 10:
                r["probe_auc_bench_labels"] = _auc(pos_b[keep_b].astype(int), s[keep_b])
                r["cv_auc_bench_labels"] = _cv_auc(H[keep_b], pos_b[keep_b].astype(int), g[keep_b], C, max_iter)
            if keep_n.sum() >= 40 and pos_n.sum() >= 10 and neg_n.sum() >= 10:
                r["probe_auc_new_labels"] = _auc(pos_n[keep_n].astype(int), s[keep_n])
                r["cv_auc_new_labels"] = _cv_auc(H[keep_n], pos_n[keep_n].astype(int), g[keep_n], C, max_iter)
            # rows where both provers agree on the task label
            both = keep_b & keep_n & (pos_b == pos_n)
            if both.sum() >= 40 and pos_b[both].sum() >= 10 and (~pos_b[both]).sum() >= 10:
                r["probe_auc_agreed_rows"] = _auc(pos_b[both].astype(int), s[both]); r["n_agreed"] = int(both.sum())
            res["tasks"][task] = r
            log.info("[%s/%s] probe AUROC bench %.3f new %.3f | CV bench %.3f new %.3f", model, task,
                     r.get("probe_auc_bench_labels") or -1, r.get("probe_auc_new_labels") or -1,
                     r.get("cv_auc_bench_labels") or -1, r.get("cv_auc_new_labels") or -1)
        out["models"][model] = res
        log.info("[%s] verdict agreement %.3f (proved/not %.3f) on %d rows", model, res["agreement"], res["proved_agreement"], len(new))
    write_json(d / "results.json", out)
    return d / "results.json"
