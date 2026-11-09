"""Qualitative failure-mode and confounder audit (data side).  Writes JSON summaries
and candidate examples under $LEANSTEER_SCRATCH/audit/ for qualitative_examples.md.

Probe decision rule used throughout: the saved logistic-regression probe at the
validation-selected layer (prompt_last anchor), decision = coef_std · ((h − mu)/sd)
+ intercept, predicted positive iff decision > 0 (direction_L<k>.npz fields)."""
from __future__ import annotations

import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leansteer.experiment import load_config, Run  # noqa: E402
from leansteer.data.faithform import load_af, build_task, problem_split, extract_statement, PROVED, REFUTED  # noqa: E402
from leansteer.run.collect import load_acts  # noqa: E402
from leansteer import config as lcfg  # noqa: E402

cfg = load_config("configs/default.yaml"); run = Run(cfg)
OUT = Path(lcfg.SCRATCH_ROOT) / "audit"; OUT.mkdir(exist_ok=True)
MODELS = ["goedel", "stepfun", "kimina", "herald"]
seed, test_frac = int(cfg["analysis"]["seed"]), cfg["analysis"]["test_frac"]

PROCEDURAL = re.compile(r"^(i |we |let's|let us|now |next |then |first|so i|to (find|solve|get|compute|do)|compute|calculate|find|solve|substitut|simplif|expand|multipl|divid|add |subtract|rewrite|plug|apply|use |using|consider|notice|note that|recall|check|try|want|need)", re.I)
def declarative(t: str) -> bool:
    return not PROCEDURAL.search(t.strip())

def feats(t: str) -> dict:
    return {"chars": len(t), "words": len(t.split()), "sentences": max(1, len(re.findall(r"[.!?](\s|$)", t))),
            "n_eq": t.count("="), "n_digits": sum(c.isdigit() for c in t), "latex": len(re.findall(r"\\[a-zA-Z]+", t)),
            "symbols": len(re.findall(r"[=<>≤≥≠±×÷√∑∏∫∞^_/*+\-]", t)), "sym_ratio": len(re.findall(r"[=<>≤≥≠±×÷√∑∏∫∞^_/*+\-\d]", t)) / max(1, len(t)),
            "declarative": int(declarative(t)), "has_therefore": int(bool(re.search(r"\b(so|therefore|thus|hence)\b", t, re.I))),
            "starts_I": int(t.strip().lower().startswith("i ")), "ends_period": int(t.strip().endswith("."))}

def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 0
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0

def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")

def decision(H, z):
    return ((H - z["mu"]) / z["sd"]) @ z["coef_std"] + float(z["intercept"])

summary = {}
for model in MODELS:
    df = load_af(models=[model], with_outputs=True).reset_index(drop=True)
    X, row_of, index = load_acts(run, model, "prompt_last"); li_of = {l: i for i, l in enumerate(index["layers"])}
    df = df[df.uid.isin(row_of)].reset_index(drop=True)
    assign = problem_split(df.problem_id, test_frac, seed)
    te = df[df.problem_id.map(assign).eq("test")].reset_index(drop=True)
    idx = np.array([row_of[u] for u in te.uid]); o = np.argsort(idx)
    F = pd.DataFrame([feats(t) for t in te.step_content])
    res = {"n_test": len(te)}
    # ---------------- Check 1: proved direction on TRUE steps (the proved task's positives are true&Proved; evaluate on true steps to avoid the corruption confound) and on all task rows
    rp = json.loads((run.probe_dir(model, "prompt_last", "proved") / "result.json").read_text()); L = rp["selected_layer"]
    z = np.load(run.probe_dir(model, "prompt_last", "proved") / f"direction_L{L}.npz")
    H = np.asarray(X[idx[o], li_of[L]]).astype(np.float32)[np.argsort(o)]
    s_proved = decision(H, z); pred = (s_proved > 0).astype(int)
    te = te.assign(s_proved=s_proved, pred_proved=pred)
    tr_rows = te[~te.is_perturbed]
    y = (tr_rows.verdict == PROVED).astype(int).to_numpy(); p = tr_rows.pred_proved.to_numpy(); sc = tr_rows.s_proved.to_numpy()
    c1 = {"layer": L, "n_true_steps": int(len(tr_rows)), "auc_true_steps": auc(y, sc), "pred_pos_rate": float(p.mean()), "label_pos_rate": float(y.mean()),
          "accuracy": float((p == y).mean()), "confounders": {}}
    Ft = F.loc[tr_rows.index]
    for c in Ft.columns:
        v = Ft[c].to_numpy(float)
        c1["confounders"][c] = {"mean_pred_pos": float(v[p == 1].mean()), "mean_pred_neg": float(v[p == 0].mean()), "d_pred": cohen_d(v[p == 1], v[p == 0]),
                                "auc_for_pred": auc(p, v), "mean_label_pos": float(v[y == 1].mean()), "mean_label_neg": float(v[y == 0].mean()), "auc_for_label": auc(y, v),
                                "corr_with_score": float(np.corrcoef(v, sc)[0, 1]) if v.std() > 0 else 0.0}
    # stratified AUROC within confounder bins
    strat = {}
    for c, bins in (("declarative", [0, 1]), ("subset", None)):
        vals = Ft[c].to_numpy() if c != "subset" else tr_rows.subset.to_numpy()
        for b in (bins if bins else sorted(set(vals))):
            m = vals == b
            if m.sum() >= 50:
                strat[f"{c}={b}"] = {"n": int(m.sum()), "auc": auc(y[m], sc[m]), "label_pos_rate": float(y[m].mean()), "pred_pos_rate": float(p[m].mean())}
    for c in ("chars", "sym_ratio", "n_eq"):
        v = Ft[c].to_numpy(float); q = np.quantile(v, [0, .33, .66, 1.0])
        for i in range(3):
            m = (v >= q[i]) & (v <= q[i + 1]) if i == 2 else (v >= q[i]) & (v < q[i + 1])
            if m.sum() >= 50:
                strat[f"{c} tercile {i+1} [{q[i]:.2f},{q[i+1]:.2f}]"] = {"n": int(m.sum()), "auc": auc(y[m], sc[m]), "label_pos_rate": float(y[m].mean()), "pred_pos_rate": float(p[m].mean())}
    c1["stratified_auc"] = strat
    # residual AUROC after regressing all surface features out of the score (linear)
    Z = np.c_[np.ones(len(Ft)), Ft.to_numpy(float), pd.get_dummies(tr_rows.subset).to_numpy(float)]
    beta, *_ = np.linalg.lstsq(Z, sc, rcond=None); resid = sc - Z @ beta
    c1["auc_after_regressing_surface_features_out_of_score"] = auc(y, resid)
    c1["surface_features_auc_for_label_(LR)"] = None
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    try:
        lr_s = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), (Z - Z.mean(0)) / (Z.std(0) + 1e-6), y, cv=5, method="decision_function")
        c1["surface_features_auc_for_label_(LR)"] = auc(y, lr_s)
    except Exception as e:
        c1["surface_features_auc_for_label_(LR)"] = str(e)
    # high-confidence disagreements
    fp = tr_rows[(y == 0)].assign(score=sc[y == 0]).sort_values("score", ascending=False).head(6)
    fn = tr_rows[(y == 1)].assign(score=sc[y == 1]).sort_values("score").head(6)
    c1["false_positives"] = [{"uid": r.uid, "score": round(r.score, 2), "verdict": r.verdict, "step": r.step_content[:220], "statement": (extract_statement(r.autoformalizer_output) or "")[:200]} for r in fp.itertuples()]
    c1["false_negatives"] = [{"uid": r.uid, "score": round(r.score, 2), "verdict": r.verdict, "step": r.step_content[:220], "statement": (extract_statement(r.autoformalizer_output) or "")[:200]} for r in fn.itertuples()]
    # matched contrast pairs: same problem, both declarative, |chars diff| <= 15, one Proved one not (Unknown/Refuted/TC), largest score gap in the *right* direction and in the wrong direction
    pairs_right, pairs_wrong = [], []
    for pid, g in tr_rows.assign(decl=Ft["declarative"].to_numpy(), chars=Ft["chars"].to_numpy()).groupby("problem_id"):
        pos = g[(g.verdict == PROVED) & (g.decl == 1)]; neg = g[(g.verdict != PROVED) & (g.decl == 1)]
        for a in pos.itertuples():
            for b in neg.itertuples():
                if abs(a.chars - b.chars) <= 15:
                    rec = {"problem_id": pid, "proved_uid": a.uid, "proved_step": a.step_content, "proved_score": round(a.s_proved, 2), "proved_statement": (extract_statement(a.autoformalizer_output) or "")[:220],
                           "other_uid": b.uid, "other_verdict": b.verdict, "other_step": b.step_content, "other_score": round(b.s_proved, 2), "other_statement": (extract_statement(b.autoformalizer_output) or "")[:220],
                           "chars": (int(a.chars), int(b.chars))}
                    (pairs_right if a.s_proved > b.s_proved else pairs_wrong).append(rec)
    c1["n_matched_pairs"] = len(pairs_right) + len(pairs_wrong); c1["matched_pairs_probe_correct_rate"] = len(pairs_right) / max(1, len(pairs_right) + len(pairs_wrong))
    c1["pairs_right_examples"] = sorted(pairs_right, key=lambda r: -(r["proved_score"] - r["other_score"]))[:5]
    c1["pairs_wrong_examples"] = sorted(pairs_wrong, key=lambda r: (r["proved_score"] - r["other_score"]))[:5]
    res["check1_proved"] = c1
    # ---------------- Check 2: sycophancy direction + abstention gate (lowest 20% of the sycophancy score among all test rows)
    rs = json.loads((run.probe_dir(model, "prompt_last", "sycophancy") / "result.json").read_text()); Ls = rs["selected_layer"]
    zs = np.load(run.probe_dir(model, "prompt_last", "sycophancy") / f"direction_L{Ls}.npz")
    Hs = np.asarray(X[idx[o], li_of[Ls]]).astype(np.float32)[np.argsort(o)]
    s_syc = Hs @ zs["direction"]; thr = np.quantile(s_syc, 0.2); rejected = s_syc <= thr
    te = te.assign(s_syc=s_syc, rejected=rejected)
    comp = lambda m: {f"{'corrupted' if p_ else 'true'}/{v}": int(n) for (p_, v), n in Counter(zip(te.is_perturbed[m], te.verdict[m])).items()}
    c2 = {"layer": Ls, "threshold_q20": float(thr), "rejected_composition": comp(rejected), "accepted_composition": comp(~rejected)}
    cor = te[te.is_perturbed]; c2["corrupted_rows"] = {"n": int(len(cor)), "rejected_rate": float(cor.rejected.mean()),
        "rejected_rate_by_verdict": {v: float(g.rejected.mean()) for v, g in cor.groupby("verdict")},
        "share_of_rejected_corrupted_that_are_Proved(sycophantic)": float((cor[cor.rejected].verdict == PROVED).mean()),
        "share_of_accepted_corrupted_that_are_Proved": float((cor[~cor.rejected].verdict == PROVED).mean())}
    tru = te[~te.is_perturbed]; c2["true_rows"] = {"n": int(len(tru)), "rejected_rate": float(tru.rejected.mean()), "rejected_rate_by_verdict": {v: float(g.rejected.mean()) for v, g in tru.groupby("verdict")}}
    # within corrupted: does the sycophancy score separate Proved (repaired) from Refuted (faithful)?
    cc = cor[cor.verdict.isin([PROVED, REFUTED])]
    c2["within_corrupted_auc_syc_score_Proved_vs_Refuted"] = auc((cc.verdict == PROVED).astype(int).to_numpy(), cc.s_syc.to_numpy())
    c2["auc_syc_score_true_vs_corrupted"] = auc((~te.is_perturbed).astype(int).to_numpy(), s_syc)
    # perturbations that did not change the maths much: near-identical text (similarity > .97) but still labelled corrupted
    import difflib
    truth_text = tru.set_index(["problem_id", "node_id"]).step_content
    sims = []
    for r in cor.itertuples():
        t = truth_text.get((r.problem_id, r.node_id))
        if isinstance(t, str):
            sims.append((r.uid, difflib.SequenceMatcher(None, t, r.step_content).ratio(), t, r.step_content, r.verdict, bool(r.rejected), round(r.s_syc, 2),
                         (extract_statement(r.autoformalizer_output) or r.autoformalizer_output or "")[:200] if r.verdict != REFUTED else "(negation proof stored)"))
    simdf = pd.DataFrame(sims, columns=["uid", "sim", "true_step", "pert_step", "verdict", "rejected", "s_syc", "artefact"])
    c2["near_noop_perturbations(sim>0.97)"] = {"n": int((simdf.sim > 0.97).sum()), "rejected_rate": float(simdf[simdf.sim > 0.97].rejected.mean()) if (simdf.sim > 0.97).any() else None,
                                                "verdicts": dict(Counter(simdf[simdf.sim > 0.97].verdict))}
    # example candidates for the refuted table: rejected corrupted Proved (sycophancy caught), rejected corrupted Refuted (faithful but rejected), rejected true Proved (false alarm), accepted corrupted Proved (missed sycophancy)
    def ex(sub, k=3):
        return [{"uid": r.uid, "sim": round(r.sim, 3), "true_step": r.true_step[:200], "pert_step": r.pert_step[:200], "verdict": r.verdict, "rejected": r.rejected, "s_syc": r.s_syc, "artefact": r.artefact} for r in sub.head(k).itertuples()]
    c2["examples"] = {
        "rejected_corrupted_Proved": ex(simdf[simdf.rejected & (simdf.verdict == PROVED)].sort_values("s_syc")),
        "rejected_corrupted_Refuted": ex(simdf[simdf.rejected & (simdf.verdict == REFUTED)].sort_values("s_syc")),
        "accepted_corrupted_Proved_missed": ex(simdf[(~simdf.rejected) & (simdf.verdict == PROVED)].sort_values("s_syc", ascending=False)),
        "near_noop_rejected": ex(simdf[(simdf.sim > 0.97) & simdf.rejected].sort_values("s_syc")),
    }
    rt = tru[tru.rejected & (tru.verdict == PROVED)].sort_values("s_syc").head(3)
    c2["examples"]["rejected_true_Proved_false_alarm"] = [{"uid": r.uid, "step": r.step_content[:200], "s_syc": round(r.s_syc, 2), "statement": (extract_statement(r.autoformalizer_output) or "")[:200]} for r in rt.itertuples()]
    res["check2_sycophancy"] = c2
    summary[model] = res
    print(f"[{model}] proved: auc(true steps) {c1['auc_true_steps']:.3f} pred_pos {c1['pred_pos_rate']:.2f} | declarative d={c1['confounders']['declarative']['d_pred']:+.2f} chars d={c1['confounders']['chars']['d_pred']:+.2f} sym_ratio d={c1['confounders']['sym_ratio']['d_pred']:+.2f} | surface-LR auc {c1['surface_features_auc_for_label_(LR)'] if isinstance(c1['surface_features_auc_for_label_(LR)'], str) else round(c1['surface_features_auc_for_label_(LR)'],3)} | score residual auc {c1['auc_after_regressing_surface_features_out_of_score']:.3f} | matched pairs {c1['n_matched_pairs']} correct {c1['matched_pairs_probe_correct_rate']:.2f}")
    print(f"[{model}] gate: rejected corrupted {c2['corrupted_rows']['rejected_rate']:.2f} by verdict {{{', '.join(f'{k}:{v:.2f}' for k,v in c2['corrupted_rows']['rejected_rate_by_verdict'].items())}}} | true rejected {c2['true_rows']['rejected_rate']:.2f} | within-corrupted Proved-vs-Refuted auc {c2['within_corrupted_auc_syc_score_Proved_vs_Refuted']:.3f} | true-vs-corrupted auc {c2['auc_syc_score_true_vs_corrupted']:.3f} | near-noop {c2['near_noop_perturbations(sim>0.97)']}")
json.dump(summary, open(OUT / "audit_checks12.json", "w"), indent=1, default=str)
print("written", OUT / "audit_checks12.json")
