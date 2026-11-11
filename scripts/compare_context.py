#!/usr/bin/env python
"""Side-by-side Goedel results: bare-step prompts (run af_v1) vs the benchmark's
<context> prompts (run af_ctx).  Writes ``context_comparison.md`` (repo root by
default).  Sections appear only when both runs have the artefact.

    python scripts/compare_context.py [--base af_v1] [--ctx af_ctx] [--model goedel] [--out context_comparison.md]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from leansteer import config

TASKS = ["proved", "proved_strict", "failing", "sycophancy", "perturbed_detect", "perturbed_proved"]
ANCHORS = ["prompt_last", "step_mean"]


FINDINGS = """## Findings (written 2026-11-11 from the tables below)

1. **The af_v1 prompts were not the benchmark's.** FaithformBench formalized
   every step with its DAG parents in a `<context>` block; af_v1 sent the bare
   step. 86 % of Goedel's steps have at least one parent and Goedel's recorded
   reasoning refers to the context in 45 % of those outputs.
2. **Context lifts every pre-generation probe whose negatives are corrupted
   steps** (5 split seeds): proved 0.73 → 0.78, proved_strict 0.78 → 0.86,
   sycophancy 0.78 → 0.86 (step_mean 0.77 → 0.89), corruption detector
   0.75 → 0.83; failing 0.67 → 0.69. Text baselines move ≤ 0.05 (TF-IDF on
   step + context), so the gain is in the representation, and the probe's
   margin over all covariates grows (proved 0.11 → 0.16, sycophancy 0.14 → 0.22).
3. **The gain is corruption detection, not provability or repair.** The proved
   direction scored on true steps only is 0.72 with or without context; within
   corrupted steps the sycophancy direction still ranks the silently repaired
   (Proved) steps as *more* corrupted-looking than the refuted ones (AUROC 0.39
   both ways). A pure corruption detector scores 0.85 on the sycophancy task
   with context. Teacher-forced probes are unchanged (0.86 / 0.92 → 0.86 / 0.94).
4. **The abstention gate does not improve**: UFLB at 20 % coverage loss
   0.197 → 0.177 with context vs → 0.166 without, because the sharper detector
   also refuses faithfully refuted corrupted steps (false alarms on true steps
   drop 8 % → 3 %, but refusals of corrupted-Refuted rise 26 % → 32 %).
5. **Steering gets weaker, not stronger.** Goedel's only significant af_v1
   effect (more refutations on corrupted steps along the proved direction,
   range +0.12, p < 0.001) shrinks to +0.05 (p = 0.047) with context; the
   sycophancy direction remains null (|range| ≤ 0.04, p ≥ 0.06), random
   directions match. Repair of corrupted steps is reproduced at α = 0 under
   the benchmark prompt (40 % Proved on corrupted-Proved rows).
6. **Regenerations match the benchmark better but not fully**: α = 0 verdict
   agreement 0.63 → 0.65 (proved rows) and 0.58 → 0.63 (sycophancy rows);
   exact statement matches 10 % → 15 %, near-duplicates (similarity > 0.9)
   17 % → 26 %. The remainder is sampling: the benchmark drew its outputs
   stochastically.

Implication for af_v1: all four models' probes, steering and prover results
there were obtained on a prompt the benchmark did not use. The qualitative
conclusions survive for Goedel (same confounder structure, same null
steering), but absolute AUROCs for proved / proved_strict / sycophancy are
underestimates of what the benchmark prompt supports, by 0.05–0.12."""


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _delta(a, b):
    return "—" if a is None or b is None else f"{b - a:+.3f}"


def probes_section(base: Path, ctx: Path, model: str, lines: list, store="probes", anchors=ANCHORS, tasks=TASKS):
    sb, sc = _load(base / model / store / "summary.json"), _load(ctx / model / store / "summary.json")
    if not sb or not sc:
        return False
    tf = store != "probes"
    title = "Teacher-forced probes" if tf else "Pre-generation probes"
    extra = "TF-IDF step+output (no ctx / ctx)" if tf else "TF-IDF step+context"
    lines += [f"### {title} (test AUROC, validation-selected layer)", "",
              f"| anchor | task | no context | context | Δ | layer (no ctx → ctx) | TF-IDF step | {extra} |",
              "|---|---|---|---|---|---|---|---|"]
    for a in anchors:
        for t in tasks:
            kb, kc = sb.get(f"{a}/{t}"), sc.get(f"{a}/{t}")
            if not kb or not kc:
                continue
            bb, bc = kb["baselines"], kc["baselines"]
            lines.append(f"| {a} | {t} | {_f(kb['selected']['test_auc'])} | {_f(kc['selected']['test_auc'])} | "
                         f"{_delta(kb['selected']['test_auc'], kc['selected']['test_auc'])} | L{kb['selected_layer']} → L{kc['selected_layer']} | "
                         f"{_f(bc.get('tfidf_auc'))} | "
                         f"{(_f(bb.get('tfidf_output_auc')) + ' / ' + _f(bc.get('tfidf_output_auc'))) if tf else _f(bc.get('tfidf_context_auc'))} |")
    lines.append("")
    return True


def uncertainty_section(base: Path, ctx: Path, model: str, lines: list):
    ub, uc = _load(base / model / "uncertainty" / "results.json"), _load(ctx / model / "uncertainty" / "results.json")
    if not ub or not uc:
        return
    lines += ["### Split-seed and bootstrap uncertainty (prompt_last)", "",
              "| task | no context: mean ± sd over 5 seeds (bootstrap 95% CI, seed 0) | context: same | Δ mean |", "|---|---|---|---|"]
    for t in ub:
        if t not in uc:
            continue
        b, c = ub[t], uc[t]
        lines.append(f"| {t} | {_f(b['mean'])} ± {_f(b['sd'])} ({_f(b['bootstrap_ci95'][0])}–{_f(b['bootstrap_ci95'][1])}) | "
                     f"{_f(c['mean'])} ± {_f(c['sd'])} ({_f(c['bootstrap_ci95'][0])}–{_f(c['bootstrap_ci95'][1])}) | {_delta(b['mean'], c['mean'])} |")
    lines.append("")


def controls_section(base: Path, ctx: Path, model: str, lines: list):
    cb, cc = _load(base / model / "controls" / "results.json"), _load(ctx / model / "controls" / "results.json")
    if not cb or not cc:
        return
    lines += ["### Perturbation controls (prompt_last; rows = direction, cols = evaluation)", "",
              "| direction | evaluation | no context | context | Δ |", "|---|---|---|---|---|"]
    for d, evs in cb["aucs"].items():
        for e, v in evs.items():
            w = cc["aucs"].get(d, {}).get(e)
            lines.append(f"| {d} | {e} | {_f(v)} | {_f(w)} | {_delta(v, w)} |")
    for k in ("sycophancy_auc_without_perturbed_dir", "sycophancy_auc_without_proved_unp_dir", "perturbed_auc_without_sycophancy_dir"):
        if k in cb and k in cc:
            lines.append(f"| {k} | — | {_f(cb[k])} | {_f(cc[k])} | {_delta(cb[k], cc[k])} |")
    lines.append("")


def abstain_section(base: Path, ctx: Path, model: str, lines: list):
    ab, ac = _load(base / model / "abstain" / "results.json"), _load(ctx / model / "abstain" / "results.json")
    if not ab or not ac:
        return
    lines += ["### Abstention gate at 20 % coverage loss (UFLB, lower is better)", "",
              "| gate | no context: UFLB (sycophantic removed) | context: UFLB (sycophantic removed) |", "|---|---|---|"]
    lines.append(f"| none | {_f(ab['base']['UFLB'])} | {_f(ac['base']['UFLB'])} |")
    for g in ab["curves"]:
        if g not in ac["curves"]:
            continue
        pb = min(ab["curves"][g], key=lambda r: abs(r["q"] - 0.2))
        pc = min(ac["curves"][g], key=lambda r: abs(r["q"] - 0.2))
        lines.append(f"| {g} | {_f(pb['UFLB'])} ({_f(pb['sycophantic_removed'], 2)}) | {_f(pc['UFLB'])} ({_f(pc['sycophantic_removed'], 2)}) |")
    lines.append("")


def baselines_section(base: Path, ctx: Path, model: str, lines: list):
    rows = []
    for t in ("proved", "proved_strict", "failing", "sycophancy"):
        b, c = _load(base / model / "baselines" / f"{t}.json"), _load(ctx / model / "baselines" / f"{t}.json")
        if b and c:
            rows.append((t, b, c))
    if not rows:
        return
    lines += ["### Covariate baselines (prompt_last; nuisance = surface features, difficulty = subset/step counts/other-step verdicts, "
              "covariates = both, oracle = adds the step's own benchmark verdict family)", "",
              "| task | probe (no ctx → ctx) | nuisance | difficulty | covariates | gain beyond covariates (no ctx → ctx) | residual probe AUROC after covariates (no ctx → ctx) |",
              "|---|---|---|---|---|---|---|"]
    for t, b, c in rows:
        lines.append(f"| {t} | {_f(b['probe_auc'])} → {_f(c['probe_auc'])} | {_f(b['nuisance_auc'])} → {_f(c['nuisance_auc'])} | "
                     f"{_f(b['difficulty_auc'])} → {_f(c['difficulty_auc'])} | {_f(b['covariates_auc'])} → {_f(c['covariates_auc'])} | "
                     f"{_f(b['probe_gain_beyond_covariates'])} → {_f(c['probe_gain_beyond_covariates'])} | "
                     f"{_f(b['residual_probe_auc_covariates'])} → {_f(c['residual_probe_auc_covariates'])} |")
    lines.append("")


def _alpha0_agreement(run: Path, model: str, task: str, prove_dir: Path):
    gen = run / model / "steer" / task / "generations.jsonl"
    ver = _load(prove_dir / "verified.json")
    if not gen.exists() or not ver:
        return None
    from leansteer.run.prove import statement_key
    agree, by_label = Counter(), defaultdict(Counter)
    for line in gen.read_text().splitlines():
        g = json.loads(line)
        if g.get("alpha") != 0 or g.get("role") != "eval":
            continue
        v = ver.get(statement_key(g["statement"])) if g.get("statement") else None
        verdict = v["verdict"] if v else "AF_Fail"
        agree[verdict == g["verdict"]] += 1
        by_label[g["label"]][verdict] += 1
    n = sum(agree.values())
    if not n:
        return None
    return {"n": n, "agreement": agree[True] / n,
            "proved_rate": {lab: c["Proved"] / max(1, sum(c.values())) for lab, c in by_label.items()}}


def prove_section(base: Path, ctx: Path, model: str, lines: list):
    pb, pc = _load(base / "prove" / "results.json"), _load(ctx / "prove" / "results.json")
    if not pb or not pc:
        return
    lines += ["### Steering + prover (DeepSeek-Prover-V2, 300 steps per class; dm direction)", ""]
    for task in ("proved", "sycophancy"):
        rows_b = [r for r in pb["table"] if r["model"] == model and r["task"] == task and r["direction"] in ("dm", "random") and r["role"] == "eval"]
        rows_c = [r for r in pc["table"] if r["model"] == model and r["task"] == task and r["direction"] in ("dm", "random") and r["role"] == "eval"]
        if not rows_b or not rows_c:
            continue
        lines += [f"**{task}** — Proved / Refuted / TC-Fail rate by label and α", "",
                  "| direction | α | label | no context P/R/TC | context P/R/TC |", "|---|---|---|---|---|"]
        key = lambda r: (r["direction"], r["alpha"], r["label"])
        mb = {key(r): r for r in rows_b}
        for r in sorted(rows_c, key=key):
            b = mb.get(key(r))
            fmt = lambda x: f"{x['proved_rate']:.2f} / {x['refuted_rate']:.2f} / {x['tc_fail_rate']:.2f}" if x else "—"
            lines.append(f"| {r['direction']} | {r['alpha']:+.0f} | {r['label']} | {fmt(b)} | {fmt(r)} |")
        lines.append("")
        tb = [t for t in pb["trends"] if t["model"] == model and t["task"] == task and t["direction"] == "dm"]
        tc = [t for t in pc["trends"] if t["model"] == model and t["task"] == task and t["direction"] == "dm"]
        if tb and tc:
            lines += ["| outcome | label | no context: range effect (perm p) | context: range effect (perm p) |", "|---|---|---|---|"]
            mt = {(t["label"], t["outcome"]): t for t in tb}
            for t in tc:
                b = mt.get((t["label"], t["outcome"]))
                lines.append(f"| {t['outcome']} | {t['label']} | {_f(b['range_effect']) if b else '—'} ({_f(b['perm_p']) if b else '—'}) | "
                             f"{_f(t['range_effect'])} ({_f(t['perm_p'])}) |")
            lines.append("")
    # α = 0 regeneration vs benchmark verdicts
    lines += ["**Unsteered regeneration (α = 0) vs the benchmark's verdict for the same step**", "",
              "| task | run | n | verdict agreement | Proved rate, label 1 | Proved rate, label 0 |", "|---|---|---|---|---|---|"]
    for task in ("proved", "sycophancy"):
        for name, run in (("no context", base), ("context", ctx)):
            a = _alpha0_agreement(run, model, task, run / "prove")
            if a:
                lines.append(f"| {task} | {name} | {a['n']} | {_f(a['agreement'])} | {_f(a['proved_rate'].get(1))} | {_f(a['proved_rate'].get(0))} |")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="af_v1"); ap.add_argument("--ctx", default="af_ctx")
    ap.add_argument("--model", default="goedel"); ap.add_argument("--out", default="context_comparison.md")
    a = ap.parse_args()
    base, ctx = config.RUNS_DIR / a.base, config.RUNS_DIR / a.ctx
    lines = [f"# {a.model}: bare-step prompts vs the benchmark's context prompts", "",
             f"Runs `{a.base}` (prompt = step only) and `{a.ctx}` (prompt = `<context>` block of the step's parent steps, then "
             "\"Based on the context, formalize the following statement: <step>\", byte-identical to FaithformBench's "
             "`DAGNode.contextualized()`).  Same problem split, probe settings and labels; only the prompt differs.  "
             "Generated by `scripts/compare_context.py`.", "", FINDINGS, ""]
    probes_section(base, ctx, a.model, lines)
    uncertainty_section(base, ctx, a.model, lines)
    controls_section(base, ctx, a.model, lines)
    abstain_section(base, ctx, a.model, lines)
    baselines_section(base, ctx, a.model, lines)
    probes_section(base, ctx, a.model, lines, store="probes_tf", anchors=["statement_mean", "output_last"], tasks=["proved_unp", "failing_unp"])
    prove_section(base, ctx, a.model, lines)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
