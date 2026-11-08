"""Stage ``probe-heads``: attention-head probes versus residual-stream probes.

Following the Inference-Time-Intervention recipe, the per-head attention
outputs (the input of each block's ``o_proj``, ``n_heads`` vectors of
``head_dim``) are probed at the same anchors, tasks and problem splits as the
residual probes.  For each (model, anchor, task):

  per-head       a logistic-regression probe on each head's ``head_dim``
                 features at every layer: validation AUROC (selection) and
                 test AUROC -> a layer x head heat-map
  per-layer      a probe on all heads of one layer concatenated (dimension
                 n_heads * head_dim = d_model), directly comparable with the
                 residual probe at the same layer
  top-k heads    the k heads with the best validation AUROC anywhere in the
                 network, concatenated (k in ``heads.top_k``)

and the residual probe's selected-layer AUROC is copied alongside.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from leansteer.analysis.probes import _Std, _auc, _fit, fit_layer
from leansteer.data.faithform import build_task, load_af, problem_split
from leansteer.experiment import Run, write_json
from leansteer.run.collect import load_acts

log = logging.getLogger(__name__)


def _head_probe(Htr, ytr, vtr, Hte, yte, C, max_iter):
    fitmask = ~vtr
    std = _Std(Htr[fitmask]); m = _fit(std(Htr[fitmask]), ytr[fitmask], C, max_iter)
    val = _auc(ytr[vtr], m.decision_function(std(Htr[vtr])))
    std = _Std(Htr); m = _fit(std(Htr), ytr, C, max_iter)
    return val, _auc(yte, m.decision_function(std(Hte)))


def run_head_probes(cfg: dict, model: str, tasks: Optional[Sequence[str]] = None,
                    anchors: Optional[Sequence[str]] = None) -> dict:
    run = Run(cfg)
    a, hc = cfg["analysis"], cfg["heads"]
    C, max_iter, seed = float(hc.get("probe_C", a["probe_C"])), int(a["max_iter"]), int(a["seed"])
    head_C = float(hc.get("head_probe_C", 1.0))
    df = load_af(models=[model])
    tasks = list(tasks or cfg["tasks"])
    anchors = list(anchors or hc["anchors"])
    summary = {}
    for anchor in anchors:
        X, row_of, index = load_acts(run, model, anchor, store="acts_heads")
        layers, H, dh = index["layers"], int(index["n_heads"]), int(index["head_dim"])
        li_of = {l: i for i, l in enumerate(layers)}
        stride = int(hc.get("layer_stride", 1))
        use = sorted(set(layers[::stride]) | {layers[-1]})
        for task in tasks:
            table = build_task(df, task, seed, balance=task in cfg["balance"])
            table = table[table.uid.isin(row_of)].reset_index(drop=True)
            if len(table) < 40 or table.label.sum() < 10 or (table.label == 0).sum() < 10:
                continue
            assign = problem_split(table.group, a["test_frac"], seed)
            te = table.group.map(assign).eq("test").to_numpy()
            tr_t, te_t = table[~te], table[te]
            vtr = tr_t.group.map(problem_split(tr_t.group, a["val_frac"], seed + 1)).eq("test").to_numpy()
            ytr, yte = tr_t.label.to_numpy(), te_t.label.to_numpy()
            idx_tr = np.array([row_of[u] for u in tr_t.uid]); idx_te = np.array([row_of[u] for u in te_t.uid])
            Xtr = np.asarray(X[np.sort(idx_tr)]).astype(np.float32)[np.argsort(np.argsort(idx_tr))]
            Xte = np.asarray(X[np.sort(idx_te)]).astype(np.float32)[np.argsort(np.argsort(idx_te))]
            log.info("[%s/%s/%s] head probes: %d layers x %d heads (n=%d)", model, anchor, task, len(use), H, len(table))
            # per head
            jobs = [(l, h) for l in use for h in range(H)]
            res = Parallel(n_jobs=int(a["n_jobs"]), prefer="threads")(
                delayed(_head_probe)(Xtr[:, li_of[l], h * dh:(h + 1) * dh], ytr, vtr, Xte[:, li_of[l], h * dh:(h + 1) * dh], yte, head_C, max_iter)
                for l, h in jobs)
            heads_df = pd.DataFrame([{"layer": l, "head": h, "val_auc": v, "test_auc": t} for (l, h), (v, t) in zip(jobs, res)])
            # per layer, all heads concatenated
            lres = Parallel(n_jobs=int(a["n_jobs"]), prefer="threads")(
                delayed(fit_layer)(Xtr[:, li_of[l]], ytr, vtr, Xte[:, li_of[l]], yte, C, max_iter) for l in use)
            layers_df = pd.DataFrame([{"layer": l, "val_auc": r["val_auc"], "test_auc": r["test_auc"], "dm_test_auc": r["dm_test_auc"]} for l, r in zip(use, lres)])
            sel_layer = int(layers_df.sort_values("val_auc").layer.iloc[-1])
            # top-k heads by validation AUROC
            ranked = heads_df.sort_values("val_auc", ascending=False)
            topk = {}
            for k in hc["top_k"]:
                pick = ranked.head(int(k))
                cols_tr = np.concatenate([Xtr[:, li_of[int(r.layer)], int(r.head) * dh:(int(r.head) + 1) * dh] for r in pick.itertuples()], axis=1)
                cols_te = np.concatenate([Xte[:, li_of[int(r.layer)], int(r.head) * dh:(int(r.head) + 1) * dh] for r in pick.itertuples()], axis=1)
                r = fit_layer(cols_tr, ytr, vtr, cols_te, yte, C if k > 4 else head_C, max_iter)
                topk[str(k)] = {"val_auc": r["val_auc"], "test_auc": r["test_auc"],
                                "heads": [f"L{int(x.layer)}H{int(x.head)}" for x in pick.itertuples()][:16]}
            best_head = ranked.iloc[0]
            # residual probe for comparison
            rp = run.probe_dir(model, anchor, task) / "result.json"
            resid = json.loads(rp.read_text()) if rp.exists() else None
            d = run.probe_dir(model, anchor, task, "probes_heads")
            heads_df.to_csv(d / "heads.csv", index=False); layers_df.to_csv(d / "layers.csv", index=False)
            result = {
                "model": model, "anchor": anchor, "task": task, "n": len(table), "n_heads": H, "head_dim": dh,
                "best_head": {"layer": int(best_head["layer"]), "head": int(best_head["head"]), "val_auc": float(best_head["val_auc"]), "test_auc": float(best_head["test_auc"])},
                "head_auc_quantiles": {q: float(heads_df.test_auc.quantile(float(q))) for q in ("0.5", "0.9", "0.99")},
                "n_heads_above_0.6": int((heads_df.test_auc > 0.6).sum()), "n_heads_total": len(heads_df),
                "all_heads_layer": {"selected_layer": sel_layer, "val_auc": float(layers_df.set_index("layer").loc[sel_layer, "val_auc"]),
                                    "test_auc": float(layers_df.set_index("layer").loc[sel_layer, "test_auc"]),
                                    "best_test_auc": float(layers_df.test_auc.max())},
                "top_k": topk,
                "residual": {"selected_layer": resid["selected_layer"], "test_auc": resid["selected"]["test_auc"],
                             "best_test_auc": resid["best_test_auc"]} if resid else None,
            }
            write_json(d / "result.json", result)
            _heatmap(heads_df, d / "heatmap.png", f"{model} {anchor} {task}: per-head test AUROC")
            summary[f"{anchor}/{task}"] = result
            log.info("[%s/%s/%s] best head L%dH%d %.3f | all-heads L%d %.3f | top-k %s | residual L%s %.3f", model, anchor, task,
                     result["best_head"]["layer"], result["best_head"]["head"], result["best_head"]["test_auc"], sel_layer,
                     result["all_heads_layer"]["test_auc"], {k: round(v["test_auc"], 3) for k, v in topk.items()},
                     resid["selected_layer"] if resid else "?", resid["selected"]["test_auc"] if resid else -1)
    write_json(run.model_dir(model) / "probes_heads" / "summary.json", summary)
    return summary


def _heatmap(heads_df: pd.DataFrame, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = heads_df.pivot(index="layer", columns="head", values="test_auc")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap="viridis", vmin=0.5, vmax=max(0.6, float(np.nanmax(M.to_numpy()))), origin="lower")
    ax.set_yticks(range(len(M.index))[::max(1, len(M.index) // 9)]); ax.set_yticklabels(M.index[::max(1, len(M.index) // 9)])
    ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label="test AUROC"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
