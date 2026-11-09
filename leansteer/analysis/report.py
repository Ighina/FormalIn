"""Stage ``report``: markdown summary and AUROC-by-layer plots."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from leansteer.experiment import Run


def _fmt(x):
    return "–" if x is None else f"{x:.3f}"


def run_report(cfg: dict) -> Path:
    import shutil
    run = Run(cfg)
    rep = run.report_dir()
    for model in cfg["models"]:
        for p in (run.model_dir(model) / "probes_heads").glob("*/*/heatmap.png"):
            shutil.copy(p, rep / f"heads_{model}_{p.parent.parent.name}_{p.parent.name}.png")
    lines = [f"# LeanSteer report — run `{cfg['run_name']}`\n"]
    probe_rows = []
    for model in cfg["models"]:
        p = run.model_dir(model) / "probes" / "summary.json"
        if not p.exists():
            continue
        for key, r in json.loads(p.read_text()).items():
            s, b = r["selected"], r["baselines"]
            probe_rows.append({"model": model, "anchor": r["anchor"], "task": r["task"], "n": r["n"], "pos": r["n_pos"],
                               "layer": r["selected_layer"], "val": s["val_auc"], "test_auroc": s["test_auc"],
                               "test_ap": s["test_ap"], "bacc": s["test_bacc"], "diffmeans": s["dm_test_auc"],
                               "best_any_layer": r["best_test_auc"], "tfidf": b["tfidf_auc"], "length": b["length_auc"]})
    if probe_rows:
        lines.append("## Probes (held-out problems; AUROC at the validation-selected layer)\n")
        lines.append("| model | anchor | task | n | pos | layer | val | **test AUROC** | AP | bal.acc | diff-means | best layer (test) | TF-IDF text | length |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in probe_rows:
            lines.append(f"| {r['model']} | {r['anchor']} | {r['task']} | {r['n']} | {r['pos']} | {r['layer']} | {_fmt(r['val'])} | "
                         f"**{_fmt(r['test_auroc'])}** | {_fmt(r['test_ap'])} | {_fmt(r['bacc'])} | {_fmt(r['diffmeans'])} | "
                         f"{_fmt(r['best_any_layer'])} | {_fmt(r['tfidf'])} | {_fmt(r['length'])} |")
        lines.append("")
        _plots(run, cfg, rep, lines)
    for model in cfg["models"]:
        for anchor in cfg["analysis"]["anchors"]:
            t = run.model_dir(model) / "probes" / anchor / "transfer.json"
            if not t.exists():
                continue
            tr = json.loads(t.read_text())
            tasks = sorted({k.split("->")[0] for k in tr})
            lines.append(f"### Cross-task transfer — {model} / {anchor} (rows: direction from, cols: evaluated on)\n")
            lines.append("| from \\ on | " + " | ".join(tasks) + " |")
            lines.append("|---|" + "---|" * len(tasks))
            for s in tasks:
                lines.append(f"| {s} (L{tr[f'{s}->{s}']['layer']}) | " + " | ".join(_fmt(tr[f"{s}->{d}"]["auc"]) for d in tasks) + " |")
            lines.append("")
    for model in cfg["models"]:
        p = run.model_dir(model) / "steer" / "summary.json"
        if not p.exists():
            continue
        for task, r in json.loads(p.read_text()).items():
            lines.append(f"## Steering — {model} / {task} (layer {r['layer']})\n")
            cols = ["direction", "alpha", "label", "n", "parse_rate", "changed_rate", "sim_base", "typecheck_rate", "think_chars"]
            if task == "sycophancy":
                cols += ["match_true_rate", "sim_true"]
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "---|" * len(cols))
            for row in r["table"]:
                lines.append("| " + " | ".join(_cell(row.get(c)) for c in cols) + " |")
            lines.append("")
    _uncertainty_section(run, cfg, lines)
    _controls_section(run, cfg, lines)
    _baselines_section(run, cfg, lines)
    _abstain_section(run, cfg, lines)
    _heads_section(run, cfg, lines)
    _tf_section(run, cfg, lines)
    _prove_section(run, cfg, lines)
    _external_section(run, cfg, lines)
    out = rep / "summary.md"
    out.write_text("\n".join(lines))
    return out


def _uncertainty_section(run: Run, cfg: dict, lines: list) -> None:
    rows = []
    for model in cfg["models"]:
        p = run.model_dir(model) / "uncertainty" / "results.json"
        if p.exists():
            for task, r in json.loads(p.read_text()).items():
                rows.append((model, task, r))
    if not rows:
        return
    lines.append("## Uncertainty of the probe AUROCs (prompt_last, selected layer)\n")
    lines.append("95% interval from a problem-clustered bootstrap of the seed-0 test set (2000 draws); mean ± sd over "
                 f"{cfg['analysis'].get('n_seeds', 5)} problem-level train/test splits; TF-IDF step-text baseline alongside.\n")
    lines.append("| model | task | layer | AUROC (seed 0) | bootstrap 95% CI | seeds mean ± sd | TF-IDF mean ± sd | TF-IDF 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model, task, r in rows:
        lines.append(f"| {model} | {task} | {r['layer']} | {_fmt(r['seeds'][0])} | [{_fmt(r['bootstrap_ci95'][0])}, {_fmt(r['bootstrap_ci95'][1])}] | "
                     f"{_fmt(r['mean'])} ± {_fmt(r['sd'])} | {_fmt(r['tfidf_mean'])} ± {_fmt(r['tfidf_sd'])} | [{_fmt(r['tfidf_bootstrap_ci95'][0])}, {_fmt(r['tfidf_bootstrap_ci95'][1])}] |")
    lines.append("")


def _controls_section(run: Run, cfg: dict, lines: list) -> None:
    rows = []
    for model in cfg["models"]:
        p = run.model_dir(model) / "controls" / "results.json"
        if p.exists():
            rows.append((model, json.loads(p.read_text())))
    if not rows:
        return
    lines.append("## Perturbation-confound controls (prompt_last, sycophancy's selected layer)\n")
    lines.append("Directions trained on train problems (columns) scored on held-out evaluation sets (rows). `within_perturbed` = corrupted steps only, "
                 "Proved vs Refuted (no perturbation confound); `perturbed` direction = a pure corruption detector; `proved_unp` direction = Proved vs not "
                 "among true steps (never sees a corrupted step).\n")
    dirs = ["proved_unp", "perturbed", "sycophancy", "proved_strict"]
    evs = ["within_perturbed", "within_perturbed_any", "within_true", "sycophancy", "proved_strict", "perturbed"]
    lines.append("| model | eval set (test) | n / pos | " + " | ".join(f"{d} dir" for d in dirs) + " |")
    lines.append("|---|---|---|" + "---|" * len(dirs))
    for model, r in rows:
        for ev in evs:
            if ev not in r["aucs"]:
                continue
            sz = r["sizes"][f"test_{ev}"]
            lines.append(f"| {model} | {ev} | {sz['n']} / {sz['pos']} | " + " | ".join(_fmt(r["aucs"][ev].get(d)) for d in dirs) + " |")
    lines.append("")
    lines.append("| model | cos(sycophancy, perturbed) | cos(sycophancy, proved_unp) | sycophancy AUROC w/o perturbed dir | w/o proved_unp dir | perturbed AUROC w/o sycophancy dir |")
    lines.append("|---|---|---|---|---|---|")
    for model, r in rows:
        c = r["cosines"]
        lines.append(f"| {model} | {_fmt(c['perturbed~sycophancy'])} | {_fmt(c['proved_unp~sycophancy'])} | {_fmt(r['sycophancy_auc_without_perturbed_dir'])} | "
                     f"{_fmt(r['sycophancy_auc_without_proved_unp_dir'])} | {_fmt(r['perturbed_auc_without_sycophancy_dir'])} |")
    lines.append("")


def _abstain_section(run: Run, cfg: dict, lines: list) -> None:
    rows = []
    for model in cfg["models"]:
        p = run.model_dir(model) / "abstain" / "results.json"
        if p.exists():
            rows.append((model, json.loads(p.read_text())))
    if not rows:
        return
    lines.append("## Abstention with the pre-generation probe (held-out problems, all step variants)\n")
    lines.append("FaithformBench's final metrics (arXiv:2608.10916): FNR = P(Refuted | true step), FPR = P(Proved | corrupted step), AFFR = P(AF_Fail or TC_Fail), "
                 "**UFLB = ½FNR + ½FPR + AFFR (lower is better)**. Rows with the lowest scores are dropped; *syc. removed* = share of the sycophantic cases "
                 "(corrupted & Proved) that were dropped.\n")
    lines.append("| model | score | abstain q | coverage | TPR | FNR | FPR | AFFR | **UFLB** | syc. removed | true removed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, r in rows:
        b = r["base"]
        lines.append(f"| {model} | (none) | 0 | 1.00 | {_fmt(b['TPR'])} | {_fmt(b['FNR'])} | {_fmt(b['FPR'])} | {_fmt(b['AFFR'])} | **{_fmt(b['UFLB'])}** | – | – |")
        for sname, curve in r["curves"].items():
            for c in curve:
                if c["q"] in (0.1, 0.2, 0.3):
                    lines.append(f"| {model} | {sname} | {c['q']} | {_fmt(c['coverage'])} | {_fmt(c['TPR'])} | {_fmt(c['FNR'])} | {_fmt(c['FPR'])} | {_fmt(c['AFFR'])} | **{_fmt(c['UFLB'])}** | "
                                 f"{_fmt(c['sycophantic_removed'])} | {_fmt(c['true_removed'])} |")
    lines.append("")


def _baselines_section(run: Run, cfg: dict, lines: list) -> None:
    rows = []
    for model in cfg["models"]:
        p = run.model_dir(model) / "baselines" / "summary.json"
        mb = run.model_dir(model) / "modernbert" / "summary.json"
        mbs = json.loads(mb.read_text()) if mb.exists() else {}
        if not p.exists():
            continue
        for task, r in json.loads(p.read_text()).items():
            rows.append({"model": model, "task": task, "layer": r["layer"], "probe": r["probe_auc"],
                         "modernbert": (mbs.get(task) or {}).get("test_auc"),
                         "nuisance": r["nuisance_auc"], "difficulty": r["difficulty_auc"], "covariates": r["covariates_auc"],
                         "oracle": r["oracle_auc"], "gain_cov": r["probe_gain_beyond_covariates"], "gain_or": r["probe_gain_beyond_oracle"],
                         "resid_cov": r["residual_probe_auc_covariates"], "resid_or": r["residual_probe_auc_oracle"],
                         "r2_cov": r["probe_score_r2_on_covariates"], "r2_or": r["probe_score_r2_on_oracle"],
                         "within": r})
    if not rows:
        return
    lines.append("## Baselines and confounds (prompt_last anchor, selected layer; held-out problems)\n")
    lines.append("Covariate groups: *nuisance* = step length/numbers/operators, position in trace, anchor entropy, TF-IDF-SVD of the "
                 "step text; *difficulty* = benchmark subset, number of steps, leave-one-out proved/failing rate of the problem's other "
                 "steps (this model and pooled over all 8 formalizers), statement_type, proof_type, length; *covariates* = both; "
                 "*oracle* = the other 7 formalizers' verdicts on this very step (encodes the step's truth — a ceiling, not a nuisance). "
                 "*gain* = AUROC of covariates+probe score minus covariates alone; *resid* = probe after ridge-regressing the group out "
                 "of the activations; *R²* = share of the probe score explained by the group.\n")
    lines.append("| model | task | probe | ModernBERT (text) | nuisance | difficulty | covariates | oracle | gain beyond cov | gain beyond oracle | resid (cov) | resid (oracle) | R² cov | R² oracle |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['model']} | {r['task']} | **{_fmt(r['probe'])}** | {_fmt(r['modernbert'])} | {_fmt(r['nuisance'])} | {_fmt(r['difficulty'])} | "
                     f"{_fmt(r['covariates'])} | {_fmt(r['oracle'])} | {_fmt(r['gain_cov'])} | {_fmt(r['gain_or'])} | {_fmt(r['resid_cov'])} | "
                     f"{_fmt(r['resid_or'])} | {_fmt(r['r2_cov'])} | {_fmt(r['r2_or'])} |")
    lines.append("")
    lines.append("### Probe AUROC within strata of the held-out set\n")
    lines.append("Rows marked *trained within* fit and test the probe inside one subset only.\n")
    subsets = ["gsm8k", "math", "olympiadbench", "omnimath"]
    lines.append("| model | task | " + " | ".join(subsets) + " | " + " | ".join(f"{s} (trained within)" for s in subsets) +
                 " | by statement_type | by n_others_proved |")
    lines.append("|---|---|" + "---|" * (2 * len(subsets) + 2))
    for r in rows:
        w = r["within"]
        ws, wt = w["within_subset"], w["within_subset_trained"]
        st = ", ".join(f"{k}:{_fmt(v['auc'])}(n={v['n']})" for k, v in w["within_statement_type"].items() if not k.startswith("_"))
        no = ", ".join(f"{k}:{_fmt(v['auc'])}(n={v['n']})" for k, v in w["within_n_others_proved"].items() if not k.startswith("_"))
        lines.append(f"| {r['model']} | {r['task']} | " + " | ".join(_fmt((ws.get(s) or {}).get("auc")) for s in subsets) + " | " +
                     " | ".join(_fmt((wt.get(s) or {}).get("auc")) for s in subsets) + f" | {st} | {no} |")
    lines.append("")
    lines.append("### Correlation of the probe score with single covariates (held-out rows)\n")
    keys = ["log_chars", "step_pos", "rel_pos", "entropy", "top_prob", "problem_proved_rate", "problem_proved_rate_all", "n_others_proved", "n_others_failing"]
    lines.append("| model | task | " + " | ".join(keys) + " |")
    lines.append("|---|---|" + "---|" * len(keys))
    for r in rows:
        c = r["within"]["probe_score_corr"]
        lines.append(f"| {r['model']} | {r['task']} | " + " | ".join(_fmt(c.get(k)) for k in keys) + " |")
    lines.append("")


def _heads_section(run: Run, cfg: dict, lines: list) -> None:
    rows = []
    for model in cfg["models"]:
        p = run.model_dir(model) / "probes_heads" / "summary.json"
        if not p.exists():
            continue
        for key, r in json.loads(p.read_text()).items():
            rows.append((model, r))
    if not rows:
        return
    ks = list(rows[0][1]["top_k"])
    lines.append("## Attention-head probes vs residual-stream probes (held-out problems)\n")
    lines.append("Per-head probes read one head's output (input of `o_proj`, head_dim features); *all heads of a layer* is the "
                 "concatenation (d_model features) at the validation-selected layer; *top-k* concatenates the k heads with the best "
                 "validation AUROC anywhere in the network. The residual probe is the §Probes result at its own selected layer.\n")
    lines.append("| model | anchor | task | heads | best single head | median / p90 / p99 head AUROC | heads > 0.6 | all heads of a layer | " +
                 " | ".join(f"top-{k}" for k in ks) + " | **residual probe** |")
    lines.append("|---|---|---|---|---|---|---|---|" + "---|" * len(ks) + "---|")
    for model, r in rows:
        bh, q, al, rs = r["best_head"], r["head_auc_quantiles"], r["all_heads_layer"], r["residual"] or {}
        lines.append(f"| {model} | {r['anchor']} | {r['task']} | {r['n_heads_total']} | L{bh['layer']}H{bh['head']} {_fmt(bh['test_auc'])} | "
                     f"{_fmt(q['0.5'])} / {_fmt(q['0.9'])} / {_fmt(q['0.99'])} | {r['n_heads_above_0.6']} | L{al['selected_layer']} {_fmt(al['test_auc'])} | " +
                     " | ".join(_fmt(r["top_k"][k]["test_auc"]) for k in ks) + f" | **L{rs.get('selected_layer', '?')} {_fmt(rs.get('test_auc'))}** |")
    lines.append("")
    for model, r in rows:
        if r["anchor"] == "prompt_last":
            lines.append(f"![{model} heads {r['task']}](heads_{model}_{r['anchor']}_{r['task']}.png)\n")


def _tf_section(run: Run, cfg: dict, lines: list) -> None:
    rows = []
    for model in cfg["models"]:
        p = run.model_dir(model) / "probes_tf" / "summary.json"
        if not p.exists():
            continue
        mbp = run.model_dir(model) / "modernbert_tf" / "summary.json"
        mb = json.loads(mbp.read_text()) if mbp.exists() else {}
        for key, r in json.loads(p.read_text()).items():
            s, b = r["selected"], r["baselines"]
            m = mb.get(r["task"]) or {}
            rows.append((model, r["anchor"], r["task"], r["n"], r["n_pos"], r["selected_layer"], s["val_auc"], s["test_auc"], s["dm_test_auc"],
                         b["tfidf_auc"], b.get("tfidf_output_auc"), m.get("test_auc_mean", m.get("test_auc")), m.get("test_auc_sd")))
    if not rows:
        return
    lines.append("## Teacher-forced anchors (unperturbed steps only; the model's own recorded output appended to the prompt)\n")
    lines.append("The last columns are text classifiers that see the step *and* the Lean theorem the model wrote (the fair text baselines here): "
                 "TF-IDF on step+output, and ModernBERT fine-tuned on step ||| statement (mean ± sd over seeds).\n")
    lines.append("| model | anchor | task | n | pos | layer | val | **test AUROC** | diff-means | TF-IDF step | TF-IDF step+output | ModernBERT step+statement |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        mbcell = (_fmt(r[11]) + (f" ± {_fmt(r[12])}" if r[12] is not None else "")) if r[11] is not None else "–"
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {_fmt(r[6])} | **{_fmt(r[7])}** | {_fmt(r[8])} | {_fmt(r[9])} | {_fmt(r[10])} | {mbcell} |")
    lines.append("")


def _prove_section(run: Run, cfg: dict, lines: list) -> None:
    for folder, title in (("prove", "residual-stream steering"), ("prove_heads", f"attention-head steering (top-{cfg['heads']['steer_k']} heads)")):
        _prove_table(run, cfg, lines, folder, title)
    for v in cfg.get("variants", {}):
        _prove_table(run, cfg, lines, f"prove_{v}", f"steering variant `{v}`")
    _relabel_section(run, cfg, lines)


def _prove_table(run: Run, cfg: dict, lines: list, folder: str, title: str) -> None:
    p = run.root / folder / "results.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    lines.append(f"## Prover verdicts — {title} (DeepSeek-Prover-V2-7B, {r['n_statements']} unique statements)\n")
    lines.append(f"Overall: {r['verdicts_overall']}\n")
    cols = ["model", "task", "direction", "alpha", "label", "role", "n", "proved_rate", "refuted_rate", "unknown_rate", "tc_fail_rate", "af_fail_rate"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for row in r["table"]:
        lines.append("| " + " | ".join(_cell(row.get(c)) for c in cols) + " |")
    lines.append("")
    if r.get("trends"):
        lines.append("### Dose-response along the diff-of-means direction\n")
        lines.append("Slope of the outcome rate on alpha (dm direction, alpha 0 included) times the alpha range; p from 2000 permutations "
                     "of alpha within each step. Random-direction rates at alpha = -8 / +8 for comparison.\n")
        lines.append("| model | task | label | direction | outcome | steps | effect over range | p | rate at min α | rate at 0 | rate at max α | random (min α) | random (max α) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in r["trends"]:
            rb = t["rate_by_alpha"]; rr = t["random_rate"]
            lo, hi = min(rb, key=float), max(rb, key=float)
            lines.append(f"| {t['model']} | {t['task']} | {t['label']} | {t.get('direction', 'dm')} | {t['outcome']} | {t['n_steps']} | {t['range_effect']:+.3f} | {t['perm_p']:.3f} | "
                         f"{_fmt(rb.get(lo))} ({lo}) | {_fmt(rb.get('0.0'))} | {_fmt(rb.get(hi))} ({hi}) | {_fmt(rr.get(min(rr, key=float))) if rr else '–'} | {_fmt(rr.get(max(rr, key=float))) if rr else '–'} |")
        lines.append("")


def _relabel_section(run: Run, cfg: dict, lines: list) -> None:
    prover = cfg["relabel"].get("prover", "goedel")
    p = run.root / f"relabel_{prover}" / "results.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    lines.append(f"## Prover robustness: held-out benchmark statements relabelled with {prover} ({cfg['relabel']['per_model']} steps per model)\n")
    lines.append("| model | n | verdict agreement | proved-vs-not agreement |")
    lines.append("|---|---|---|---|")
    for m, res in r["models"].items():
        lines.append(f"| {m} | {res['n']} | {_fmt(res['agreement'])} | {_fmt(res['proved_agreement'])} |")
    lines.append("")
    lines.append("Probe AUROC on the same rows against the benchmark labels (DeepSeek-Prover-V2) and the new labels; "
                 "*fixed* = the existing direction trained on benchmark labels, *CV* = 5-fold grouped cross-validation at the selected layer "
                 "trained on each label source; *agreed* = fixed direction on rows where both provers give the same task label.\n")
    lines.append("| model | task | layer | n bench / new | fixed: bench | fixed: new | CV: bench | CV: new | fixed on agreed rows (n) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m, res in r["models"].items():
        for task, t in res["tasks"].items():
            lines.append(f"| {m} | {task} | {t['layer']} | {t['n_bench']} / {t['n_new']} | {_fmt(t.get('probe_auc_bench_labels'))} | "
                         f"{_fmt(t.get('probe_auc_new_labels'))} | {_fmt(t.get('cv_auc_bench_labels'))} | {_fmt(t.get('cv_auc_new_labels'))} | "
                         f"{_fmt(t.get('probe_auc_agreed_rows'))} ({t.get('n_agreed', '–')}) |")
    lines.append("")
    lines.append("### Verdict confusion (rows: benchmark verdict, cols: new verdict)\n")
    for m, res in r["models"].items():
        cols = sorted({c for row in res["confusion"].values() for c in row})
        lines.append(f"**{m}**\n")
        lines.append("| bench \\ new | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for k, row in res["confusion"].items():
            lines.append(f"| {k} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
        lines.append("")


def _cell(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def _plots(run: Run, cfg: dict, rep: Path, lines: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for model in cfg["models"]:
        for anchor in cfg["analysis"]["anchors"]:
            frames = []
            for task in cfg["tasks"]:
                p = run.model_dir(model) / "probes" / anchor / task / "layers.csv"
                if p.exists():
                    df = pd.read_csv(p); df["task"] = task; frames.append(df)
            if not frames:
                continue
            df = pd.concat(frames)
            fig, ax = plt.subplots(figsize=(6, 3.4))
            for task, g in df.groupby("task"):
                ax.plot(g.layer, g.test_auc, marker="o", ms=3, label=task)
            ax.axhline(0.5, color="grey", lw=0.8, ls="--")
            ax.set_xlabel("layer"); ax.set_ylabel("test AUROC"); ax.set_title(f"{model} — {anchor}"); ax.legend(fontsize=7)
            fig.tight_layout()
            name = f"auroc_{model}_{anchor}.png"
            fig.savefig(rep / name, dpi=130); plt.close(fig)
            lines.append(f"![{model} {anchor}]({name})\n")


def _external_section(run: Run, cfg: dict, lines: list) -> None:
    for name in ("prm800k", "prm800k_claims", "minif2f", "minif2f_ext"):
        ep = run.root / f"ext_{name}" / "eval.json"
        if not ep.exists():
            continue
        e = json.loads(ep.read_text())
        pp = run.root / f"ext_{name}" / "prove" / "results.json"
        lines.append(f"## Generalization: {name} ({e['n']} steps, {e['n_true']} true)\n")
        if pp.exists():
            p = json.loads(pp.read_text())
            lines.append("Our prover pass over the four models' autoformalizations (DeepSeek-Prover-V2-7B, benchmark protocol):\n")
            lines.append("| model | TPR | FNR | FPR | AFFR | **UFLB** | true: Proved / Refuted / Unknown / TC_Fail / AF_Fail | false: Proved / Refuted / TC_Fail |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in p["table"]:
                lines.append(f"| {r['model']} | {_fmt(r['TPR'])} | {_fmt(r.get('FNR'))} | {_fmt(r['FPR'])} | {_fmt(r.get('AFFR'))} | **{_fmt(r.get('UFLB'))}** | {_fmt(r['true_proved'])} / {_fmt(r['true_refuted'])} / "
                             f"{_fmt(r['true_unknown'])} / {_fmt(r['true_tc_fail'])} / {_fmt(r['true_af_fail'])} | {_fmt(r['false_proved'])} / {_fmt(r['false_refuted'])} / {_fmt(r['false_tc_fail'])} |")
            lines.append("")
        lines.append("Truth (true vs false step) from the prompt_last state: FaithformBench directions applied without retraining, a probe retrained "
                     "on this dataset (5-fold problem-grouped CV, same layer), the best layer of a CV sweep (optimistic), and TF-IDF text (CV).\n")
        lines.append("| model | corruption-detector dir | sycophancy dir | proved_strict dir | proved dir | retrained CV (same layer) | best-layer CV | TF-IDF CV |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m, r in e["models"].items():
            t = r["truth"]
            def g(prefix):
                return next((v for k, v in t.items() if k.startswith(prefix)), None)
            bl = t.get("best_layer_cv", {})
            lines.append(f"| {m} | {_fmt(g('perturbed_detect_dir'))} | {_fmt(g('sycophancy_dir'))} | {_fmt(g('proved_strict_dir'))} | {_fmt(g('proved_dir'))} | "
                         f"{_fmt(g('retrained_cv'))} | {_fmt(bl.get('auc'))} (L{bl.get('layer', '?')}) | {_fmt(t.get('tfidf_cv'))} |")
        lines.append("")
        lines.append("Verdict tasks on our prover's verdicts: FaithformBench direction without retraining / retrained CV / TF-IDF CV.\n")
        lines.append("| model | task | n (pos) | bench direction | retrained CV | TF-IDF CV |")
        lines.append("|---|---|---|---|---|---|")
        for m, r in e["models"].items():
            for task, x in r["verdict_tasks"].items():
                lines.append(f"| {m} | {task} | {x['n']} ({x['pos']}) | {_fmt(x.get('bench_dir_auc'))} | {_fmt(x.get('retrained_cv_auc'))} | {_fmt(x.get('tfidf_cv'))} |")
        lines.append("")
