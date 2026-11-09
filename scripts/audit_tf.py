"""Check 3: teacher-forced (own-output appended) probes partitioned by output-failure category."""
import json
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leansteer.experiment import load_config, Run
from leansteer.data.faithform import load_af, problem_split, extract_statement, PROVED, FAILING
from leansteer import config as lcfg
CFG = os.environ.get("AUDIT_CONFIG", "configs/default.yaml")
cfg = load_config(CFG); run = Run(cfg); seed = int(cfg["analysis"]["seed"])
OUT = Path(lcfg.SCRATCH_ROOT) / "audit" / cfg["run_name"] if cfg["run_name"] != "af_v1" else Path(lcfg.SCRATCH_ROOT) / "audit"; OUT.mkdir(parents=True, exist_ok=True)
def auc(y, s): return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
res = {}
for model in cfg["models"]:
    df = load_af(models=[model], with_outputs=True)
    df = df[~df.is_perturbed].drop_duplicates("uid").set_index("uid")
    acts = run.model_dir(model) / "acts_tf"; index = json.loads((acts / "index.json").read_text())
    meta = pd.DataFrame([json.loads(l) for l in (acts / "meta.jsonl").read_text().splitlines()]).set_index("uid")
    row_of = {u: i for i, u in enumerate(index["uids"])}; li = {l: i for i, l in enumerate(index["layers"])}
    skipped = set(index["skipped"])
    sub = df.loc[[u for u in df.index if u in row_of]].copy()
    assign = problem_split(sub.problem_id, cfg["analysis"]["test_frac"], seed)
    te = sub[sub.problem_id.map(assign).eq("test")].copy()
    out_txt = te.autoformalizer_output.fillna("")
    te["has_statement"] = meta.loc[te.index, "has_statement"].to_numpy()
    te["n_tokens"] = meta.loc[te.index, "n_tokens"].to_numpy()
    te["failed_marker"] = out_txt.str.contains("Failed to formalize")
    te["no_lean_block"] = ~out_txt.str.contains("theorem")
    te["unclosed_think"] = out_txt.str.contains("<think>") & ~out_txt.str.contains("</think>")
    te["failure"] = te.failed_marker | te.no_lean_block | (~te.has_statement) | te.unclosed_think
    r = {"n_test": int(len(te)), "n_skipped_too_long(all rows)": len(skipped), "failure_rate": float(te.failure.mean()),
         "failure_breakdown": {"failed_marker": int(te.failed_marker.sum()), "no_theorem_in_output": int(te.no_lean_block.sum()), "no_statement_span": int((~te.has_statement).sum()), "unclosed_think": int(te.unclosed_think.sum())},
         "verdict_by_failure": {str(k): dict(v.verdict.value_counts()) for k, v in te.groupby("failure")}, "tasks": {}}
    for anchor in ("statement_mean", "output_last"):
        X = np.load(acts / f"{anchor}.npy", mmap_mode="r")
        for task, pos_fn, neg_fn in (("proved_unp", lambda d: d.verdict == PROVED, lambda d: d.verdict != PROVED),
                                      ("failing_unp", lambda d: ~d.verdict.isin(FAILING), lambda d: d.verdict.isin(FAILING))):
            rp = run.probe_dir(model, anchor, task, "probes_tf") / "result.json"
            if not rp.exists(): continue
            L = json.loads(rp.read_text())["selected_layer"]; z = np.load(run.probe_dir(model, anchor, task, "probes_tf") / f"direction_L{L}.npz")
            idx = np.array([row_of[u] for u in te.index]); o = np.argsort(idx)
            H = np.asarray(X[idx[o], li[L]]).astype(np.float32)[np.argsort(o)]
            s = ((H - z["mu"]) / z["sd"]) @ z["coef_std"] + float(z["intercept"])
            y = pos_fn(te).to_numpy().astype(int); keep = (pos_fn(te) | neg_fn(te)).to_numpy()
            f = te.failure.to_numpy()
            # prompt-side (non-teacher) comparison: same task label on the prompt_last store
            Xp, row_p, idxp = None, None, None
            try:
                from leansteer.run.collect import load_acts
                Xp, row_p, indp = load_acts(run, model, "prompt_last")
                rpp = run.probe_dir(model, "prompt_last", "proved" if task == "proved_unp" else "failing") / "result.json"
                Lp = json.loads(rpp.read_text())["selected_layer"]; zp = np.load(run.probe_dir(model, "prompt_last", "proved" if task == "proved_unp" else "failing") / f"direction_L{Lp}.npz")
                lip = {l: i for i, l in enumerate(indp["layers"])}; ip = np.array([row_p[u] for u in te.index]); op = np.argsort(ip)
                Hp = np.asarray(Xp[ip[op], lip[Lp]]).astype(np.float32)[np.argsort(op)]; sp = Hp @ zp["direction"]
            except Exception as e:
                sp = None
            rec = {"layer": L, "auc_all": auc(y[keep], s[keep]), "auc_no_failure": auc(y[keep & ~f], s[keep & ~f]), "auc_failure_only": auc(y[keep & f], s[keep & f]),
                   "n_no_failure": int((keep & ~f).sum()), "n_failure": int((keep & f).sum()), "pos_rate_no_failure": float(y[keep & ~f].mean()), "pos_rate_failure": float(y[keep & f].mean()) if (keep & f).any() else None,
                   "auc_failure_flag_as_predictor": auc(y[keep], (~f[keep]).astype(float)),
                   "prompt_side_probe_auc_all": auc(y[keep], sp[keep]) if sp is not None else None, "prompt_side_probe_auc_no_failure": auc(y[keep & ~f], sp[keep & ~f]) if sp is not None else None}
            r["tasks"][f"{anchor}/{task}"] = rec
            print(f"[{model}] {anchor}/{task}: all {rec['auc_all']:.3f} | no-failure {rec['auc_no_failure']:.3f} (n={rec['n_no_failure']}) | failure-only {rec['auc_failure_only']:.3f} (n={rec['n_failure']}) | failure flag alone {rec['auc_failure_flag_as_predictor']:.3f} | prompt-side probe all {rec['prompt_side_probe_auc_all']:.3f} no-failure {rec['prompt_side_probe_auc_no_failure']:.3f}")
    print(f"[{model}] failure rate {r['failure_rate']:.3f} breakdown {r['failure_breakdown']} skipped-too-long {r['n_skipped_too_long(all rows)']}")
    res[model] = r
json.dump(res, open(OUT / "audit_tf.json", "w"), indent=1, default=str)
