"""Stage ``controls``: is the sycophancy / proved_strict direction a *perturbation
detector*?  In both tasks every negative is a corrupted step, so a probe could
in principle read the style of the LLM corruption rather than the step's
truth or provability.  Four directions are trained at the prompt_last anchor
and scored on evaluation sets that break the confound:

  directions (train problems)
    proved_unp     true steps only: Proved vs any other verdict (no perturbation label)
    perturbed      corrupted vs true, all verdicts, balanced (a pure perturbation detector)
    sycophancy     the task's own direction (true-Proved vs corrupted-Proved)
    proved_strict  true-Proved vs corrupted-Refuted

  evaluation sets (test problems)
    within_perturbed      corrupted steps only: Proved vs Refuted   (no perturbation confound)
    within_perturbed_any  corrupted steps only: Proved vs not Proved
    within_true           true steps only: Proved vs not Proved
    sycophancy            true-Proved vs corrupted-Proved
    proved_strict         true-Proved vs corrupted-Refuted
    perturbed             corrupted vs true (all verdicts)

Also reported: the sycophancy probe refitted on activations with the
perturbation-detector direction projected out, and the perturbation detector
refitted with the sycophancy direction projected out.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from leansteer.analysis.probes import _Std, _auc, _fit
from leansteer.data.faithform import FAILING, PROVED, REFUTED, build_task, load_af, problem_split
from leansteer.experiment import Run, write_json
from leansteer.run.collect import load_acts

log = logging.getLogger(__name__)


def _sets(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """name -> (positive mask, negative mask) over df rows."""
    unp, pert, pr, rf = ~df.is_perturbed.to_numpy(), df.is_perturbed.to_numpy(), (df.verdict == PROVED).to_numpy(), (df.verdict == REFUTED).to_numpy()
    return {
        "proved_unp": (unp & pr, unp & ~pr),
        "perturbed": (pert, unp),
        "sycophancy": (unp & pr, pert & pr),
        "proved_strict": (unp & pr, pert & rf),
        "within_perturbed": (pert & pr, pert & rf),
        "within_perturbed_any": (pert & pr, pert & ~pr),
        "within_true": (unp & pr, unp & ~pr),
    }


def _balanced(pos: np.ndarray, neg: np.ndarray, rng, cap: int | None = None) -> np.ndarray:
    p, n = np.flatnonzero(pos), np.flatnonzero(neg)
    k = min(len(p), len(n)) if cap is None else min(len(p), len(n), cap)
    return np.sort(np.concatenate([rng.choice(p, k, replace=False), rng.choice(n, k, replace=False)]))


def run_controls(cfg: dict, model: str, anchor: str = "prompt_last") -> dict:
    run = Run(cfg)
    a = cfg["analysis"]; C, max_iter, seed = float(a["probe_C"]), int(a["max_iter"]), int(a["seed"])
    rng = np.random.default_rng(seed)
    df = load_af(models=[model]).reset_index(drop=True)
    X, row_of, index = load_acts(run, model, anchor)
    li_of = {l: i for i, l in enumerate(index["layers"])}
    df = df[df.uid.isin(row_of)].reset_index(drop=True)
    assign = problem_split(df.problem_id, a["test_frac"], seed)
    is_te = df.problem_id.map(assign).eq("test").to_numpy()
    # layer: the sycophancy probe's selected layer (same for all directions, so they are comparable)
    layer = int(json.loads((run.probe_dir(model, anchor, "sycophancy") / "result.json").read_text())["selected_layer"])
    idx = np.array([row_of[u] for u in df.uid]); o = np.argsort(idx)
    H = np.asarray(X[idx[o], li_of[layer]]).astype(np.float32)[np.argsort(o)]
    sets = _sets(df)
    dirs: Dict[str, Tuple[np.ndarray, float]] = {}
    out = {"model": model, "anchor": anchor, "layer": layer, "n_rows": len(df), "aucs": {}, "sizes": {}}
    for name in ("proved_unp", "perturbed", "sycophancy", "proved_strict"):
        pos, neg = sets[name]
        tr = _balanced(pos & ~is_te, neg & ~is_te, rng)
        y = pos[tr].astype(int)
        std = _Std(H[tr]); clf = _fit(std(H[tr]), y, C, max_iter)
        w = clf.coef_[0] / std.sd; w /= np.linalg.norm(w) + 1e-9
        dirs[name] = (w.astype(np.float32), float(clf.intercept_[0]))
        out["sizes"][f"train_{name}"] = int(len(tr))
    for ev, (pos, neg) in sets.items():
        m = (pos | neg) & is_te
        y = pos[m].astype(int)
        out["sizes"][f"test_{ev}"] = {"n": int(m.sum()), "pos": int(y.sum())}
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            continue
        out["aucs"][ev] = {d: _auc(y, H[m] @ w) for d, (w, _) in dirs.items()}
    # cosine between directions
    names = list(dirs)
    out["cosines"] = {f"{p}~{q}": float(dirs[p][0] @ dirs[q][0]) for i, p in enumerate(names) for q in names[i + 1:]}
    # sycophancy with the perturbation direction projected out, and the converse
    def refit_without(name: str, remove: str) -> float | None:
        v = dirs[remove][0]
        Hp = H - np.outer(H @ v, v)
        pos, neg = sets[name]
        tr = _balanced(pos & ~is_te, neg & ~is_te, rng); m = (pos | neg) & is_te
        std = _Std(Hp[tr]); clf = _fit(std(Hp[tr]), pos[tr].astype(int), C, max_iter)
        return _auc(pos[m].astype(int), clf.decision_function(std(Hp[m])))
    out["sycophancy_auc_without_perturbed_dir"] = refit_without("sycophancy", "perturbed")
    out["sycophancy_auc_without_proved_unp_dir"] = refit_without("sycophancy", "proved_unp")
    out["perturbed_auc_without_sycophancy_dir"] = refit_without("perturbed", "sycophancy")
    out["within_perturbed_auc_without_perturbed_dir"] = None
    d = run.model_dir(model) / "controls"; d.mkdir(exist_ok=True)
    write_json(d / "results.json", out)
    log.info("[%s] L%d | within_perturbed (Proved vs Refuted among corrupted): proved_unp-dir %.3f sycophancy-dir %.3f perturbed-dir %.3f | "
             "sycophancy set: proved_unp-dir %.3f perturbed-dir %.3f own %.3f | sycophancy w/o perturbed-dir %.3f | cos(syc,pert) %.2f cos(syc,proved_unp) %.2f",
             model, layer, out["aucs"]["within_perturbed"]["proved_unp"], out["aucs"]["within_perturbed"]["sycophancy"],
             out["aucs"]["within_perturbed"]["perturbed"], out["aucs"]["sycophancy"]["proved_unp"], out["aucs"]["sycophancy"]["perturbed"],
             out["aucs"]["sycophancy"]["sycophancy"], out["sycophancy_auc_without_perturbed_dir"],
             out["cosines"]["perturbed~sycophancy"], out["proved_unp~sycophancy" if False else "cosines"]["proved_unp~sycophancy"])
    return out
