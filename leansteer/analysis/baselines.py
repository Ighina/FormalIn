"""Stage ``baselines``: what do the probes read that cheap covariates do not?

Mirrors REVISE's nuisance analysis.  For every (model, task) at the
``prompt_last`` anchor and the probe's selected layer:

  nuisance features   step length (chars, words, tokens), numbers / operators /
                      equation signs in the step, position of the step in its
                      trace, number of steps, benchmark subset (gsm8k, math,
                      olympiadbench, omnimath), the benchmark's step metadata
                      (statement_type, proof_type), next-token entropy and top
                      probability of the model at the anchor, and a TF-IDF-SVD
                      embedding of the step text
  difficulty features problem-level and step-type covariates that do not look at
                      this particular step's verdicts: the benchmark subset, the
                      number of steps, the leave-one-out proved rate of the
                      *other* steps of the same problem (for this model, and
                      pooled over all eight formalizers), statement_type,
                      proof_type and step length
  oracle features     the verdicts of the *other* seven formalizers on this very
                      step variant (3 dedicated + 4 LLM; proved / refuted /
                      failing indicators).  These encode whether the step is
                      true and formalizable — the signal itself, not a
                      nuisance — so they are reported separately as a ceiling,
                      and the probe is additionally evaluated *within* strata
                      of "how many other formalizers proved this step"

and reports, per covariate group G in {nuisance, difficulty, covariates =
nuisance ∪ difficulty, oracle}:

  G-only AUROC (logistic regression); stacked (G + out-of-fold probe score)
  AUROC and the gain over G; residualised probe (G ridge-regressed out of the
  activations); R² of the probe score on G
  within-stratum AUROC of the probe: inside each subset, each statement_type,
      and each bucket of n_others_proved
  within-subset training (train and test inside one benchmark subset)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from leansteer import config
from leansteer.analysis.probes import _Std, _auc, _fit, fit_layer, load_direction
from leansteer.data.faithform import AF_MODELS, FAILING, PROVED, REFUTED, build_task, load_af, normalise_verdict, problem_split
from leansteer.experiment import Run, read_jsonl, write_json
from leansteer.run.collect import load_acts

log = logging.getLogger(__name__)
LLM_MODELS = ("gpt-5.2", "gemini-3.1-pro", "claude-opus-4-7", "qwen-plus")
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
STMT_TYPES = ("computation", "logical", "substitution", "general_knowledge", "simplification", "verification")
PROOF_TYPES = ("deduction", "assumption", "axiom")


# --------------------------------------------------------------------------- features
def step_metadata() -> pd.DataFrame:
    m = pd.read_csv(config.FAITHFORM_ROOT / "FaithformBench.csv",
                    usecols=["problem_id", "node_id", "is_perturbed", "statement_type", "proof_type"], low_memory=False)
    m["is_perturbed"] = m.is_perturbed.astype(bool)
    m["uid"] = [f"{p}|{n}|{int(x)}" for p, n, x in zip(m.problem_id, m.node_id, m.is_perturbed)]
    return m.drop_duplicates("uid").set_index("uid")[["statement_type", "proof_type"]]


def other_verdicts(df_all: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per uid: proved/refuted/failing indicators of the other 3 dedicated and the
    4 LLM formalizers, plus the count of other formalizers that proved it."""
    llm = pd.read_csv(config.FAITHFORM_ROOT / "Faithform-LLM.csv",
                      usecols=["problem_id", "node_id", "is_perturbed", "autoformalizer_model", "proof_result", "step_content"], low_memory=False)
    llm["is_perturbed"] = llm.is_perturbed.astype(bool)
    llm["step_content"] = llm.step_content.astype(str).str.strip()
    llm["verdict"] = llm.proof_result.map(normalise_verdict)
    # the LLM file renumbered its nodes: join on (problem_id, step_content) like RLLF does
    key = df_all.drop_duplicates("uid")[["uid", "problem_id", "step_content"]]
    llm = llm.merge(key, on=["problem_id", "step_content"], how="inner")
    af = df_all[df_all.autoformalizer_model != model][["uid", "autoformalizer_model", "verdict"]]
    both = pd.concat([af, llm[["uid", "autoformalizer_model", "verdict"]]])
    cols = {}
    for m in [x for x in AF_MODELS if x != model] + list(LLM_MODELS):
        v = both[both.autoformalizer_model == m].drop_duplicates("uid").set_index("uid").verdict
        cols[f"{m}_proved"] = (v == PROVED).astype(float)
        cols[f"{m}_refuted"] = (v == REFUTED).astype(float)
        cols[f"{m}_failing"] = v.isin(FAILING).astype(float)
    F = pd.DataFrame(cols).reindex(df_all.uid.unique()).fillna(0.0)
    F["n_others_proved"] = F[[c for c in F.columns if c.endswith("_proved")]].sum(1)
    F["n_others_failing"] = F[[c for c in F.columns if c.endswith("_failing")]].sum(1)
    return F


def build_features(df_m: pd.DataFrame, df_all: pd.DataFrame, model: str, meta_rows: List[dict]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Returns (features indexed by uid, {group: columns})."""
    d = df_m.drop_duplicates("uid").set_index("uid")
    F = pd.DataFrame(index=d.index)
    s = d.step_content
    F["log_chars"] = np.log1p(s.str.len()); F["log_words"] = np.log1p(s.str.split().str.len())
    F["n_numbers"] = s.str.count(r"\d+(?:\.\d+)?"); F["n_ops"] = s.str.count(r"[+\-*/^=<>≤≥≠×÷]")
    F["n_eq"] = s.str.count("="); F["has_frac"] = s.str.contains(r"/|\\frac").astype(float)
    F["has_latex"] = s.str.contains(r"\\\\|\$").astype(float); F["n_words_alpha"] = s.str.count(r"[A-Za-z]{3,}")
    pos = d.node_id.str.extract(r"(\d+)")[0].astype(float)
    n_steps = d.groupby("problem_id").node_id.transform("nunique").astype(float)
    F["step_pos"] = pos; F["n_steps"] = n_steps; F["rel_pos"] = pos / n_steps.clip(lower=1)
    for sub in SUBSETS:
        F[f"subset_{sub}"] = (d.subset == sub).astype(float)
    md = step_metadata().reindex(d.index)
    for t in STMT_TYPES:
        F[f"stype_{t}"] = (md.statement_type == t).astype(float)
    for t in PROOF_TYPES:
        F[f"ptype_{t}"] = (md.proof_type == t).astype(float)
    m = pd.DataFrame(meta_rows).drop_duplicates("uid").set_index("uid").reindex(d.index)
    F["entropy"] = m.entropy.fillna(m.entropy.median()); F["top_prob"] = m.top_prob.fillna(m.top_prob.median())
    F["log_prompt_len"] = np.log1p(m.prompt_len.fillna(m.prompt_len.median()))
    nuisance = [c for c in F.columns if not c.startswith(("subset_", "stype_", "ptype_")) and c not in ("n_steps",)]
    O = other_verdicts(df_all, model).reindex(d.index).fillna(0.0)
    # leave-one-out proved rate of the *other* true steps of the same problem (this model)
    F["problem_proved_rate"] = _loo_problem_rate(d, d.verdict == PROVED)
    # same, pooled over all eight formalizers (dedicated + LLM)
    F["problem_proved_rate_all"] = _loo_problem_rate_all(d, df_all, O)
    F["problem_failing_rate"] = _loo_problem_rate(d, d.verdict.isin(FAILING))
    F = pd.concat([F, O], axis=1)
    groups = {
        "nuisance": nuisance,
        "difficulty": ["problem_proved_rate", "problem_proved_rate_all", "problem_failing_rate", "n_steps", "log_chars", "log_words",
                       "n_numbers"] + [c for c in F.columns if c.startswith(("subset_", "stype_", "ptype_"))],
        "oracle": list(O.columns),
    }
    groups["covariates"] = sorted(set(groups["nuisance"]) | set(groups["difficulty"]))
    return F, groups


def _loo_problem_rate(d: pd.DataFrame, flag: pd.Series) -> pd.Series:
    """Per row: mean of ``flag`` over the *other* unperturbed steps of the same problem."""
    unp = d[~d.is_perturbed]
    f = flag[unp.index].astype(float)
    tot, cnt = f.groupby(unp.problem_id).transform("sum"), f.groupby(unp.problem_id).transform("count")
    loo_unp = (tot - f) / (cnt - 1).clip(lower=1)
    # perturbed rows: rate over all unperturbed steps of the problem except their own true counterpart
    per_problem_tot = f.groupby(unp.problem_id).sum(); per_problem_cnt = f.groupby(unp.problem_id).count()
    out = pd.Series(index=d.index, dtype=float)
    out[unp.index] = loo_unp
    pert = d[d.is_perturbed]
    own = pd.MultiIndex.from_arrays([pert.problem_id, pert.node_id])
    truth = unp.set_index([unp.problem_id, unp.node_id])
    own_flag = flag[unp.index].astype(float).set_axis(truth.index).reindex(own).to_numpy()
    own_flag = np.nan_to_num(own_flag, nan=0.0)
    tot_p = per_problem_tot.reindex(pert.problem_id).to_numpy(); cnt_p = per_problem_cnt.reindex(pert.problem_id).to_numpy()
    out[pert.index] = (tot_p - own_flag) / np.clip(cnt_p - 1, 1, None)
    return out.fillna(out.mean())


def _loo_problem_rate_all(d: pd.DataFrame, df_all: pd.DataFrame, O: pd.DataFrame) -> pd.Series:
    """Leave-one-step-out proved rate of the problem's true steps pooled over all formalizers."""
    unp_all = df_all[~df_all.is_perturbed]
    rate = (unp_all.verdict == PROVED).astype(float).groupby([unp_all.problem_id, unp_all.node_id]).mean()  # per step, over 4 AF models
    per_step = rate.groupby(level=0)
    tot, cnt = per_step.transform("sum"), per_step.transform("count")
    loo = ((tot - rate) / (cnt - 1).clip(lower=1))
    key = pd.MultiIndex.from_arrays([d.problem_id, d.node_id])
    return pd.Series(loo.reindex(key).to_numpy(), index=d.index).fillna(float(rate.mean()))


def _svd_embed(train_text: pd.Series, k: int = 50):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit(train_text)
    M = vec.transform(train_text)
    svd = TruncatedSVD(n_components=min(k, M.shape[1] - 1), random_state=0).fit(M)
    return lambda texts: svd.transform(vec.transform(texts)).astype(np.float32)


# --------------------------------------------------------------------------- analysis
def _oof_scores(X, y, groups, C, max_iter, n_splits=5):
    s = np.zeros(len(y), dtype=np.float32)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        std = _Std(X[tr]); s[va] = _fit(std(X[tr]), y[tr], C, max_iter).decision_function(std(X[va]))
    return s


def _strata_auc(y: np.ndarray, s: np.ndarray, strata: np.ndarray, min_n: int = 30) -> Dict[str, dict]:
    out = {}
    for v in sorted(set(strata.tolist())):
        m = strata == v
        if m.sum() >= min_n and len(np.unique(y[m])) == 2:
            out[str(v)] = {"n": int(m.sum()), "pos_rate": float(y[m].mean()), "auc": _auc(y[m], s[m])}
    vals = [r["auc"] for r in out.values() if r["auc"] is not None]
    out["_weighted_mean"] = float(np.average(vals, weights=[r["n"] for r in out.values() if r["auc"] is not None])) if vals else None
    return out


def run_baselines(cfg: dict, model: str, anchor: str = "prompt_last") -> dict:
    run = Run(cfg)
    a = cfg["analysis"]; C, max_iter, seed = float(a["probe_C"]), int(a["max_iter"]), int(a["seed"])
    df_all = load_af()
    df_m = df_all[df_all.autoformalizer_model == model].reset_index(drop=True)
    X, row_of, index = load_acts(run, model, anchor)
    li_of = {l: i for i, l in enumerate(index["layers"])}
    meta_rows = read_jsonl(run.acts_dir(model) / "meta.jsonl")
    F, groups = build_features(df_m, df_all, model, meta_rows)
    out_dir = run.model_dir(model) / "baselines"; out_dir.mkdir(exist_ok=True)
    F.to_parquet(out_dir / "features.parquet")
    summary = {}
    for task in cfg["tasks"]:
        pd_ = run.probe_dir(model, anchor, task)
        if not (pd_ / "result.json").exists():
            continue
        layer, z = load_direction(run, model, anchor, task)
        table = build_task(df_m, task, seed, balance=task in cfg["balance"])
        table = table[table.uid.isin(row_of)].reset_index(drop=True)
        assign = problem_split(table.group, a["test_frac"], seed)
        te = table.group.map(assign).eq("test").to_numpy()
        tr_t, te_t = table[~te], table[te]
        ytr, yte = tr_t.label.to_numpy(), te_t.label.to_numpy()
        gtr = tr_t.group.to_numpy()
        embed = _svd_embed(tr_t.step_content)
        Ftr, Fte = F.reindex(tr_t.uid), F.reindex(te_t.uid)
        Etr, Ete = embed(tr_t.step_content), embed(te_t.step_content)
        Z = {}
        for g, cols in groups.items():
            Ztr, Zte = Ftr[cols].to_numpy(np.float32), Fte[cols].to_numpy(np.float32)
            if g in ("nuisance", "covariates"):   # text embedding belongs to the nuisance group
                Ztr, Zte = np.c_[Ztr, Etr], np.c_[Zte, Ete]
            Z[g] = (Ztr, Zte)
        idx_tr = np.array([row_of[u] for u in tr_t.uid]); idx_te = np.array([row_of[u] for u in te_t.uid])
        Htr = np.asarray(X[np.sort(idx_tr), li_of[layer]]).astype(np.float32)[np.argsort(np.argsort(idx_tr))]
        Hte = np.asarray(X[np.sort(idx_te), li_of[layer]]).astype(np.float32)[np.argsort(np.argsort(idx_te))]
        res: dict = {"task": task, "layer": layer, "n_train": len(tr_t), "n_test": len(te_t)}

        def lr_auc(Ztr, Zte):
            std = _Std(Ztr); m = _fit(std(Ztr), ytr, 1.0, 2000)
            return _auc(yte, m.decision_function(std(Zte))), m

        for g, (Ztr, Zte) in Z.items():
            res[f"{g}_auc"], _ = lr_auc(Ztr, Zte)
        # probe at the selected layer (refit here so train scores are out-of-fold)
        std = _Std(Htr); probe = _fit(std(Htr), ytr, C, max_iter)
        s_te = probe.decision_function(std(Hte)); s_tr_oof = _oof_scores(Htr, ytr, gtr, C, max_iter)
        res["probe_auc"] = _auc(yte, s_te)
        for name, (Ztr, Zte) in Z.items():
            ss = _Std(s_tr_oof[:, None])
            auc_stack, _ = lr_auc(np.c_[Ztr, ss(s_tr_oof[:, None])], np.c_[Zte, ss(s_te[:, None])])
            base = res[f"{name}_auc"]
            res[f"stacked_{name}_auc"] = auc_stack
            res[f"probe_gain_beyond_{name}"] = (auc_stack - base) if (auc_stack is not None and base is not None) else None
            # residualised probe: regress covariates out of the activations
            zs = _Std(Ztr); ridge = Ridge(alpha=1.0).fit(zs(Ztr), Htr)
            Rtr, Rte = Htr - ridge.predict(zs(Ztr)), Hte - ridge.predict(zs(Zte))
            rs = _Std(Rtr); res[f"residual_probe_auc_{name}"] = _auc(yte, _fit(rs(Rtr), ytr, C, max_iter).decision_function(rs(Rte)))
            # how much of the probe score do the covariates explain?
            r2 = Ridge(alpha=1.0).fit(zs(Ztr), s_tr_oof).score(zs(Zte), s_te)
            res[f"probe_score_r2_on_{name}"] = float(r2)
        # correlations of the probe score with single covariates (test rows)
        corr = {}
        for c in ["log_chars", "step_pos", "rel_pos", "entropy", "top_prob", "problem_proved_rate", "problem_proved_rate_all",
                  "n_others_proved", "n_others_failing"]:
            v = Fte[c].to_numpy(np.float32)
            corr[c] = float(np.corrcoef(v, s_te)[0, 1]) if v.std() > 0 else None
        res["probe_score_corr"] = corr
        # within-stratum AUROC on test rows
        res["within_subset"] = _strata_auc(yte, s_te, te_t.subset.to_numpy())
        res["within_statement_type"] = _strata_auc(yte, s_te, step_metadata().reindex(te_t.uid).statement_type.fillna("?").to_numpy())
        res["within_n_others_proved"] = _strata_auc(yte, s_te, Fte["n_others_proved"].to_numpy().astype(int))
        res["within_label_balance_by_subset"] = {s: float(yte[te_t.subset.to_numpy() == s].mean()) for s in SUBSETS if (te_t.subset == s).any()}
        # train *and* test inside one subset
        within = {}
        for sub in SUBSETS:
            mtr, mte = tr_t.subset.to_numpy() == sub, te_t.subset.to_numpy() == sub
            if mtr.sum() >= 200 and mte.sum() >= 50 and len(np.unique(ytr[mtr])) == 2 and len(np.unique(yte[mte])) == 2:
                r = fit_layer(Htr[mtr], ytr[mtr], np.zeros(mtr.sum(), bool), Hte[mte], yte[mte], C, max_iter)
                within[sub] = {"n_train": int(mtr.sum()), "n_test": int(mte.sum()), "auc": r["test_auc"]}
        res["within_subset_trained"] = within
        summary[task] = res
        write_json(out_dir / f"{task}.json", res)
        log.info("[%s/%s] probe %.3f | nuisance %.3f difficulty %.3f covariates %.3f oracle %.3f | gain beyond covariates %.3f, "
                 "beyond oracle %.3f | residual(covariates) %.3f residual(oracle) %.3f | R2(covariates) %.2f R2(oracle) %.2f",
                 model, task, res["probe_auc"], res["nuisance_auc"], res["difficulty_auc"], res["covariates_auc"], res["oracle_auc"],
                 res["probe_gain_beyond_covariates"], res["probe_gain_beyond_oracle"], res["residual_probe_auc_covariates"],
                 res["residual_probe_auc_oracle"], res["probe_score_r2_on_covariates"], res["probe_score_r2_on_oracle"])
    write_json(out_dir / "summary.json", summary)
    return summary
