"""Build background.md: the report companion.  Every number is read from the run
artefacts under $LEANSTEER_SCRATCH/runs/<run>/ so the document can be regenerated
and each table traced to a file.  Usage:
    PYTHONPATH=. python scripts/build_background.py [configs/default.yaml] > background.md
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leansteer.experiment import load_config  # noqa: E402

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")
from leansteer import config as lcfg  # noqa: E402
R = lcfg.RUNS_DIR / cfg["run_name"]
MODELS = ["goedel", "stepfun", "kimina", "herald"]
TASKS = ["proved", "proved_strict", "failing", "sycophancy"]


def J(*parts):
    p = R.joinpath(*parts)
    return json.loads(p.read_text()) if p.exists() else None


def f(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def tbl(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def verify(items):
    """items: list of (label, text)."""
    return "\n".join(f"- **{k}:** {v}" for k, v in items) + "\n"


out = []
P = out.append

# ----------------------------------------------------------------------------- header
P(f"""# LeanSteer — background and verification guide

This document is the companion to the report: it states each result, gives the
numbers as they stand in the run artefacts, says which code produced them and
where the files are, and how to recompute them. It is **generated** by
`scripts/build_background.py` from the run `{cfg['run_name']}`; rerun it after
any stage and the tables update. Run root: `{R}`. Human-written interpretation
is marked as such; everything in a table comes from a JSON file named in the
"verify" box below it.

Conventions: AUROC on held-out *problems* (30 % of problem ids, seed {cfg['analysis']['seed']}),
probe layer chosen on a validation split of the training problems, never on
test. "text" = TF-IDF (1–2-grams) logistic regression on the step text with the
same split. Model keys: goedel = Goedel-Formalizer-V2-8B, stepfun =
StepFun-Formalizer-7B, kimina = Kimina-Autoformalizer-7B, herald =
Herald_translator.

## 0. How to verify anything here

- Regenerate the full report with plots: `python -m leansteer.cli report -c configs/default.yaml`
  → `{R}/report/summary.md`.
- Every stage is idempotent and resumable; rerunning a stage on existing
  artefacts recomputes its JSON without touching others. GPU stages on this
  host go through `scripts/gpuq_stage.sh`, CPU stages through `scripts/cpu_stage.sh`
  (see README §Running).
- Unit tests (`python -m pytest -q`, 13 tests) cover the task definitions,
  statement extraction, anchor capture, the steering hooks and tokenizer
  round-trips.
- Data file: `{lcfg.FAITHFORM_ROOT}/Faithform-AF.csv`; loader
  `leansteer/data/faithform.py::load_af`; the task tables are printed by
  `python -m leansteer.cli data -c configs/default.yaml`.

""")

# ----------------------------------------------------------------------------- 1 question + data
P("""## 1. Question and data

**TL;DR.** FaithformBench-AF autoformalizes true and corrupted reasoning steps
with four models and scores the result with a prover. We ask whether the
prover's verdict is already encoded in the autoformalizer's residual stream
while it reads the step, whether that encoding is a text/difficulty artefact,
whether it is causally usable, and whether it generalizes.

### 1.1 The benchmark file (`leansteer/data/faithform.py`)

`Faithform-AF.csv`: one row per (step variant, model), 111,464 rows; columns
used: `problem_id`, `node_id`, `is_perturbed`, `step_content`,
`autoformalizer_model`, `proof_result` (Proved, Refuted, Unknown, TC-Fail,
AF-Fail, Declarative), and `autoformalizer_output` / `prover_output`.
Three facts shaped the design (verified on the data):

- Declarative rows were never autoformalized (no output) → dropped.
- `autoformalizer_output` holds the raw model response only for true steps;
  for corrupted steps it holds prover artefacts (proof, negation proof,
  statement with `sorry`, or `-- Failed to formalize`) → the main probes read
  the model *before* generation.
- 30 corrupted steps are textually identical to their true step → dropped;
  every corrupted node has a true counterpart with the same `(problem_id, node_id)`.

### 1.2 Tasks (`faithform.py::TASKS`, `build_task`)

""")
P(tbl(["task", "positives", "negatives", "note"], [
    ["`proved`", "true step, Proved", "everything else", "negatives subsampled to the positive count (`balance`)"],
    ["`proved_strict`", "true step, Proved", "corrupted step, Refuted", ""],
    ["`failing`", "verdict ∉ {AF-Fail, TC-Fail}", "AF-Fail or TC-Fail", "Unknown counts as not failing"],
    ["`sycophancy`", "true step, Proved", "corrupted step, Proved", "the benchmark's sycophantic case"],
    ["`perturbed_detect`", "true step (any verdict)", "corrupted step (any verdict)", "control: a pure corruption detector; balanced"],
    ["`perturbed_proved`", "corrupted & Proved", "corrupted & Refuted", "control: sycophancy among corrupted steps only"],
    ["`proved_unp`, `failing_unp`", "as proved / failing", "restricted to true steps", "for teacher-forced anchors (§4.4)"],
]))
try:
    from leansteer.data.faithform import load_af, task_summary
    ts = task_summary(load_af(), cfg["balance"], cfg["analysis"]["seed"])
    rows = [[r.model, r.task, r.n, r.n_pos, r.n_neg, r.n_problems] for r in ts.itertuples() if r.task in TASKS + ["perturbed_detect", "perturbed_proved"]]
    P("Task sizes (rows / positives / negatives / problems):\n\n" + tbl(["model", "task", "n", "pos", "neg", "problems"], rows))
except Exception as e:  # pragma: no cover
    P(f"_task sizes unavailable: {e}_\n")
P(verify([("code", "`leansteer/data/faithform.py` (`load_af`, `TASKS`, `build_task`, `problem_split`, `extract_statement`)"),
          ("recompute", "`python -m leansteer.cli data -c configs/default.yaml`"),
          ("tests", "`tests/test_faithform.py`")]))

# ----------------------------------------------------------------------------- 2 pipeline map
P("""## 2. Pipeline map

**TL;DR.** One YAML config, one CLI, resumable stages, one scratch tree per run.

""")
P(tbl(["stage (`python -m leansteer.cli …`)", "code", "writes (under run root)", "section"], [
    ["`collect`", "`run/collect.py::run_collect`", "`<model>/acts/{prompt_last,step_last,step_mean}.npy`, `index.json`, `meta.jsonl`", "§3"],
    ["`probe`", "`analysis/probes.py::run_probes`", "`<model>/probes/<anchor>/<task>/{result.json,layers.csv,direction_L*.npz}`, `transfer.json`, `summary.json`", "§4"],
    ["`uncertainty`", "`analysis/uncertainty.py`", "`<model>/uncertainty/results.json`", "§4.2"],
    ["`controls`", "`analysis/controls.py`", "`<model>/controls/results.json`", "§5.2"],
    ["`baselines`", "`analysis/baselines.py`", "`<model>/baselines/{<task>.json,summary.json,features.parquet}`", "§5.1"],
    ["`modernbert`, `modernbert-tf`", "`analysis/modernbert.py`", "`<model>/modernbert{,_tf}/summary.json`", "§5.3, §4.4"],
    ["`collect-heads`, `probe-heads`", "`run/collect.py::run_collect_heads`, `analysis/heads.py`", "`<model>/acts_heads/`, `<model>/probes_heads/…/{result.json,heads.csv,heatmap.png}`", "§6"],
    ["`collect-tf`, `probe-tf`", "`run/collect.py::run_collect_tf`, `probes.py` (store=acts_tf)", "`<model>/acts_tf/`, `<model>/probes_tf/`", "§4.4"],
    ["`abstain`", "`analysis/abstain.py`", "`<model>/abstain/results.json`", "§7"],
    ["`steer`, `steer-heads`, `steer-variant`", "`run/steer.py`, `run/variants.py`", "`<model>/steer{,_heads,_<variant>}/<task>/{generations.jsonl,results.json}`", "§8"],
    ["`prove-gen`, `prove-verify` (+ `-heads-`, `-variant-`)", "`run/prove.py`, `analysis/lean.py`", "`prove{,_heads,_<variant>}/{statements.json,candidates.jsonl,verified.json,results.json}`", "§8"],
    ["`relabel-select/gen/verify/eval`", "`analysis/relabel.py`, `run/prove.py`", "`relabel_goedel/{rows.json,statements.json,verified.json,results.json}`", "§9"],
    ["`ext-collect/generate/prove-gen/prove-verify/eval`", "`data/external.py`, `run/external.py`, `analysis/external_eval.py`", "`ext_<name>/{steps.parquet,<model>/acts,<model>/generations.jsonl,prove/,verdicts.json,eval.json}`", "§10"],
    ["`report`", "`analysis/report.py`", "`report/summary.md`, plots", "all"],
]))

# ----------------------------------------------------------------------------- 3 models / backend
P("""## 3. Models, prompts and what is read

**TL;DR.** The benchmark's verbatim prompts; a hooked HF backend capturing the
residual stream after every block at weighted token positions ("anchors") in
one prefill, plus additive / projection / per-head interventions for steering.

- Prompts: `leansteer/models/registry.py` (copied from the benchmark's
  `veriform/formalization/fine_tuned.py`; Goedel's statement abuts "Think
  before…" with no separator; StepFun gets a literal `<think>` appended).
  **Caveat found 2026-11-09:** the benchmark filled the template with
  `DAGNode.contextualized()` — a `<context>` block of the step's parent steps
  followed by "Based on the context, formalize the following statement:
  <step>" — while run `af_v1` filled it with the bare step. `data.context:
  true` (`configs/context.yaml`, run `af_ctx`) reproduces the benchmark text
  byte for byte (`faithform.py::contextualize`); §12 measures the effect.
- Tokenizers: Herald's (and DeepSeek-Prover's) repository class resolves to a
  slow, space-dropping tokenizer under transformers ≥ 5;
  `backend.py::load_tokenizer` forces the fast backend and asserts round-trip
  (`tests/test_backend.py::test_tokenizer_roundtrip`). An earlier Herald run
  made without this guard was discarded.
- Anchors (`run/collect.py::anchor_weights`): `prompt_last` (last prompt
  token), `step_last`, `step_mean` (mean over the step's tokens, found via
  offset mappings). Stores are float16 memmaps `[N, n_layers+1, d]`; layer
  `l` = residual after block `l`, layer 0 = embeddings.
- Interventions (`backend.py::Intervention`): `steer` (add at anchor onward /
  all / generated-only positions), `project` (set the component along a unit
  direction to a target), `head_steer` (add per-head vectors to the input of
  `o_proj`). Hook correctness is tested on a tiny model
  (`tests/test_backend.py`).

""")

# ----------------------------------------------------------------------------- 4 probes
P("""## 4. Probes

**TL;DR.** Pre-generation probes read `proved`, `proved_strict` and
`sycophancy` at AUROC 0.66–0.80 (text 0.54–0.62), with ±0.02 intervals; one
shared direction across those three tasks; `failing` sits at the text
baseline. Post-output (teacher-forced) anchors read 0.82–0.96, but so does a
text model that sees the output.

### 4.1 Pre-generation probes (prompt_last and step_mean anchors)

""")
rows = []
for m in MODELS:
    s = J(m, "probes", "summary.json") or {}
    for t in TASKS:
        for anc in ("prompt_last", "step_mean"):
            r = s.get(f"{anc}/{t}")
            if r:
                rows.append([m, anc, t, r["n"], r["n_pos"], r["selected_layer"], f(r["selected"]["val_auc"], 3), f"**{f(r['selected']['test_auc'], 3)}**",
                             f(r["selected"]["dm_test_auc"], 3), f(r["baselines"]["tfidf_auc"], 3), f(r["baselines"]["length_auc"], 3)])
P(tbl(["model", "anchor", "task", "n", "pos", "layer", "val", "test AUROC", "diff-means", "TF-IDF", "length"], rows))
P(verify([("files", "`<model>/probes/<anchor>/<task>/result.json` (`selected`, `baselines`), per-layer `layers.csv`, directions `direction_L<k>.npz`"),
          ("code", "`leansteer/analysis/probes.py::run_probes`, `fit_layer`, `text_baselines`"),
          ("recompute", "`python -m leansteer.cli probe -c configs/default.yaml --models <m>` (CPU, ~5 min per model)"),
          ("plots", "`report/auroc_<model>_<anchor>.png`")]))

P("### 4.2 Uncertainty\n\nProblem-clustered bootstrap (2000 draws over held-out problems) of the seed-0 test AUROC and refits over 5 problem splits at the selected layer.\n\n")
rows = []
for m in MODELS:
    u = J(m, "uncertainty", "results.json") or {}
    for t in TASKS:
        r = u.get(t)
        if r:
            rows.append([m, t, r["layer"], f(r["seeds"][0], 3), f"[{f(r['bootstrap_ci95'][0], 3)}, {f(r['bootstrap_ci95'][1], 3)}]",
                         f"{f(r['mean'], 3)} ± {f(r['sd'], 3)}", f"{f(r['tfidf_mean'], 3)} ± {f(r['tfidf_sd'], 3)}"])
P(tbl(["model", "task", "layer", "AUROC seed 0", "bootstrap 95% CI", "5 seeds mean ± sd", "TF-IDF mean ± sd"], rows))
P(verify([("file", "`<model>/uncertainty/results.json`"), ("code", "`leansteer/analysis/uncertainty.py::run_uncertainty`, `cluster_bootstrap_auc`"),
          ("recompute", "`python -m leansteer.cli uncertainty -c configs/default.yaml --models <m>`")]))

P("### 4.3 Cross-task transfer (prompt_last; direction of the row task scored on the column task's test rows)\n\n")
for m in MODELS:
    tr = J(m, "probes", "prompt_last", "transfer.json")
    if not tr:
        continue
    tasks = [t for t in TASKS + ["perturbed_detect", "perturbed_proved"] if f"{t}->{t}" in tr]
    P(f"**{m}**\n\n" + tbl(["from \\ on"] + tasks, [[f"{s} (L{tr[f'{s}->{s}']['layer']})"] + [f(tr[f'{s}->{d}']['auc'], 2) for d in tasks] for s in tasks]))
P(verify([("file", "`<model>/probes/prompt_last/transfer.json`"), ("code", "`probes.py::cross_task_transfer`"), ("recompute", "`python -m leansteer.cli transfer -c configs/default.yaml --models <m>`")]))
P("_Interpretation (human):_ `proved`, `proved_strict` and `sycophancy` transfer to each other within a few hundredths — one shared direction; `failing` neither transfers nor receives transfer.\n\n")

P("### 4.4 Teacher-forced anchors (true steps; the model's recorded output appended to the prompt)\n\n")
rows = []
for m in MODELS:
    s = J(m, "probes_tf", "summary.json") or {}; mb = J(m, "modernbert_tf", "summary.json") or {}
    for t in ("proved_unp", "failing_unp"):
        for anc in ("output_last", "statement_mean"):
            r = s.get(f"{anc}/{t}")
            if r:
                x = mb.get(t) or {}
                rows.append([m, anc, t, r["n"], r["selected_layer"], f"**{f(r['selected']['test_auc'], 3)}**", f(r["baselines"]["tfidf_auc"], 3),
                             f(r["baselines"].get("tfidf_output_auc"), 3), (f"{f(x.get('test_auc_mean'), 3)} ± {f(x.get('test_auc_sd'), 3)}" if x else "–")])
P(tbl(["model", "anchor", "task", "n", "layer", "probe", "TF-IDF step", "TF-IDF step+output", "ModernBERT step+statement (3 seeds)"], rows))
P(verify([("files", "`<model>/probes_tf/<anchor>/<task>/result.json`, `<model>/modernbert_tf/summary.json`"),
          ("code", "`run/collect.py::run_collect_tf` (anchors `output_last`, `statement_last`, `statement_mean`; outputs > 3072 tokens skipped), `analysis/modernbert.py` with `inputs=\"step_statement\"`"),
          ("recompute", "`collect-tf` (GPU) → `probe-tf` → `modernbert-tf` (GPU)")]))
P("_Interpretation (human):_ once the formalization is written the verdict is largely in the text — a classifier that sees the step and the Lean theorem matches the post-output probe for every model. The pre-generation signal is what the activations add.\n\n")

# ----------------------------------------------------------------------------- 5 confounds
P("""## 5. Is the probe reading text, difficulty, or perturbation?

**TL;DR.** Not text or difficulty: nuisance and problem-level covariates reach
0.53–0.72 on the verdict tasks, explain ≤ 13 % of the probe score, and the
probe keeps 0.65–0.76 after they are regressed out; AUROC is flat across
subsets and step types. `failing` is a problem-level property. But the
`sycophancy`/`proved_strict` probes largely read *corruption/truth*: a pure
corruption detector reads the state at 0.69–0.80 (text 0.54), separates the
sycophancy rows as well as the sycophancy probe, and inside corrupted steps
nothing predicts which get sycophantically proved (0.56–0.69 ≈ text).

### 5.1 Covariate groups (`analysis/baselines.py`)

""")
rows = []
for m in MODELS:
    b = J(m, "baselines", "summary.json") or {}; mb = J(m, "modernbert", "summary.json") or {}
    for t in TASKS:
        r = b.get(t)
        if r:
            x = mb.get(t) or {}
            rows.append([m, t, f"**{f(r['probe_auc'], 3)}**", f(x.get("test_auc_mean", x.get("test_auc")), 3), f(r["nuisance_auc"], 3), f(r["difficulty_auc"], 3), f(r["covariates_auc"], 3),
                         f(r["oracle_auc"], 3), f"{r['probe_gain_beyond_covariates']:+.3f}", f(r["residual_probe_auc_covariates"], 3), f(r["probe_score_r2_on_covariates"], 2),
                         f(r["within_n_others_proved"].get("_weighted_mean"), 3)])
P(tbl(["model", "task", "probe", "ModernBERT (text)", "nuisance", "difficulty", "covariates", "oracle (7 other formalizers)", "gain beyond cov.", "probe after regressing cov. out", "R² score on cov.", "within n_others_proved strata"], rows))
P("""Groups: *nuisance* = length, numbers, operators, position in trace, anchor
entropy/top-prob, TF-IDF-SVD(50) of the step; *difficulty* = subset, number of
steps, leave-one-out proved/failing rate of the problem's other steps (this
model and pooled over all 8 formalizers), statement_type, proof_type;
*oracle* = the other 3 dedicated + 4 LLM formalizers' verdicts on the same
step (encodes the step's truth; reported as a ceiling, not residualised). Full
per-task JSON also holds within-subset AUROC (and within-subset *training*),
within statement_type, and correlations with single covariates.

""")
P(verify([("files", "`<model>/baselines/<task>.json`, `summary.json`; features in `features.parquet`"),
          ("code", "`analysis/baselines.py::build_features` (groups), `run_baselines` (stacking, residualisation, strata)"),
          ("metadata inputs", "`FaithformBench.csv` (statement_type, proof_type), `Faithform-LLM.csv` (LLM formalizer verdicts, joined on (problem_id, step_content))"),
          ("recompute", "`python -m leansteer.cli baselines -c configs/default.yaml --models <m>`")]))

P("### 5.2 Perturbation-confound controls (`analysis/controls.py`; prompt_last, sycophancy's selected layer)\n\n")
rows = []
for m in MODELS:
    c = J(m, "controls", "results.json")
    if not c:
        continue
    a = c["aucs"]
    rows.append([m, c["layer"], f(a["within_perturbed"]["proved_unp"], 2), f(a["within_perturbed"]["sycophancy"], 2), f(a["within_perturbed"]["perturbed"], 2),
                 f(a["sycophancy"]["proved_unp"], 2), f(a["sycophancy"]["perturbed"], 2), f(a["sycophancy"]["sycophancy"], 2),
                 f(c["cosines"]["perturbed~sycophancy"], 2), f(c["sycophancy_auc_without_perturbed_dir"], 2)])
P(tbl(["model", "layer", "within corrupted (Proved vs Refuted): proved_unp dir", "… sycophancy dir", "… perturbed dir", "sycophancy rows: proved_unp dir", "… perturbed dir", "… own dir", "cos(sycophancy, perturbed)", "sycophancy AUROC w/o perturbed dir"], rows))
rows = []
for m in MODELS:
    s = J(m, "probes", "summary.json") or {}
    pd_, pp, sy = s.get("prompt_last/perturbed_detect"), s.get("prompt_last/perturbed_proved"), s.get("prompt_last/sycophancy")
    if pd_ and pp:
        rows.append([m, f"{f(pd_['selected']['test_auc'], 3)} (text {f(pd_['baselines']['tfidf_auc'], 3)})", f"{f(pp['selected']['test_auc'], 3)} (text {f(pp['baselines']['tfidf_auc'], 3)})", f(sy["selected"]["test_auc"], 3)])
P("Per-layer probes trained *for* the control distinctions:\n\n" + tbl(["model", "corruption detector `perturbed_detect`", "among corrupted: Proved vs Refuted `perturbed_proved`", "`sycophancy` (reference)"], rows))
P(verify([("files", "`<model>/controls/results.json`; `<model>/probes/prompt_last/{perturbed_detect,perturbed_proved}/result.json`"),
          ("code", "`analysis/controls.py::run_controls` (directions `proved_unp`, `perturbed`, `sycophancy`, `proved_strict`; eval sets incl. `within_perturbed`)"),
          ("recompute", "`controls` and `probe --tasks perturbed_proved perturbed_detect`")]))
P("""_Interpretation (human):_ the corruption detector separates the sycophancy
rows as well as the sycophancy probe (sign flipped) and is anti-aligned with
it (cos −0.3 in 3.5–4k dimensions); inside corrupted steps no pre-trained
direction predicts which ones get sycophantically proved (0.39–0.54, chance or
inverted), and a probe trained for that question reaches 0.56–0.69 against
text at 0.59–0.64. The supportable claim is "the model knows the step is
false" (and, among true steps, whether it is provable), not "the model knows
it is about to repair the error".

### 5.3 ModernBERT on the step text (3 seeds)

""")
rows = []
for m in MODELS:
    mb = J(m, "modernbert", "summary.json") or {}
    rows.append([m] + [f"{f(mb[t].get('test_auc_mean'), 3)} ± {f(mb[t].get('test_auc_sd'), 3)}" if t in mb else "–" for t in TASKS])
P(tbl(["model"] + TASKS, rows))
P(verify([("file", "`<model>/modernbert/summary.json`"), ("code", "`analysis/modernbert.py` (answerdotai/ModernBERT-base, 3 epochs, lr 3e-5, max 160 tokens, class-weighted BCE, checkpoint on validation AUROC)"),
          ("recompute", "`modernbert` (GPU) with `--set modernbert.seeds=3`")]))

# ----------------------------------------------------------------------------- 6 heads
P("""## 6. Attention-head probes

**TL;DR.** One well-chosen late head matches the whole residual probe; the
signal is sparse (median head near chance); concatenating 4–64 heads beats the
residual probe by 0.05–0.11 on the verdict tasks. `failing` heads sit
mid-network and stay at the text/difficulty baselines.

""")
rows = []
for m in MODELS:
    s = J(m, "probes_heads", "summary.json") or {}
    for t in TASKS:
        r = s.get(f"prompt_last/{t}")
        if r:
            bh, al, tk, rs, q = r["best_head"], r["all_heads_layer"], r["top_k"], r["residual"] or {}, r["head_auc_quantiles"]
            rows.append([m, t, f"L{bh['layer']}H{bh['head']} {f(bh['test_auc'], 3)}", f"{f(q['0.5'], 2)} / {f(q['0.9'], 2)}", f"{r['n_heads_above_0.6']}/{r['n_heads_total']}",
                         f"L{al['selected_layer']} {f(al['test_auc'], 3)}", f(tk["4"]["test_auc"], 3), f(tk["16"]["test_auc"], 3), f(tk["64"]["test_auc"], 3), f"**L{rs.get('selected_layer')} {f(rs.get('test_auc'), 3)}**"])
P(tbl(["model", "task", "best single head", "median / p90 head AUROC", "heads > 0.6", "all heads of a layer", "top-4", "top-16", "top-64", "residual probe"], rows))
P(verify([("files", "`<model>/probes_heads/prompt_last/<task>/{result.json,heads.csv,layers.csv,heatmap.png}`"),
          ("code", "`run/collect.py::run_collect_heads` (pre-hook on `o_proj`; `backend.capture(component=\"heads\")`), `analysis/heads.py::run_head_probes`"),
          ("recompute", "`collect-heads` (GPU, ~2 min per model) → `probe-heads` (CPU)"), ("plots", "`report/heads_<model>_prompt_last_<task>.png`")]))

# ----------------------------------------------------------------------------- 7 abstention
P("""## 7. Abstention: the applied use of the probe

**TL;DR.** Refusing the 20 % of steps the sycophancy direction scores lowest
lowers the benchmark's UFLB (unfaithfulness lower bound, lower is better) for
every model and removes 40–54 % of sycophantic cases (random 21 %, a text
corruption detector 21–24 %).

""")
rows = []
for m in MODELS:
    r = J(m, "abstain", "results.json")
    if not r:
        continue
    b = r["base"]
    def cell(s, q):
        c = [x for x in r["curves"][s] if x["q"] == q][0]
        return f"{f(c['UFLB'], 3)} ({c['sycophantic_removed']:.0%})"
    rows.append([m, f"{f(b['UFLB'], 3)} (FNR {f(b['FNR'])}, FPR {f(b['FPR'])}, AFFR {f(b['AFFR'])})", cell("sycophancy_dir", 0.2), cell("sycophancy_dir", 0.3), cell("proved_dir", 0.2), cell("detect_dir", 0.2), cell("tfidf_detect", 0.2), cell("random", 0.2)])
P(tbl(["model", "no abstention", "sycophancy dir q=0.2", "sycophancy dir q=0.3", "proved dir q=0.2", "corruption detector q=0.2", "TF-IDF detector q=0.2", "random q=0.2"], rows))
P("Cells: UFLB on the remaining held-out rows (share of sycophantic cases removed). UFLB = ½FNR + ½FPR + AFFR over all step variants of the held-out problems, using the benchmark's own verdicts (FaithformBench final metric, arXiv:2608.10916; lower is better).\n\n")
P(verify([("file", "`<model>/abstain/results.json` (curves for q ∈ {0, .05, .1, .2, .3, .5})"), ("code", "`analysis/abstain.py::run_abstain`, `metrics`"),
          ("recompute", "`python -m leansteer.cli abstain -c configs/default.yaml --models <m>`")]))

# ----------------------------------------------------------------------------- 8 steering
P("""## 8. Steering, judged by the prover

**TL;DR.** Adding the class-mean difference to the residual stream is
causally effective for Kimina's provability (true steps proved 0.37 → 0.61,
type-check failures 0.32 → 0.04 across α = −8…+8, permutation p < 0.001,
random direction flat) and changes how decisively Goedel formalizes false
steps; it does nothing to sycophancy in any model (300 steps per class,
minimal detectable effect ≈ 0.11). Head steering, other layers, multi-layer,
chain-of-thought-only, ITI σ-scaled heads and projection ablation do not
change that: no intervention lowers UFLB on the sycophancy rows beyond its
random control.

### 8.1 Protocol

- Generation: `run/steer.py::run_steer` — `α · |μ₊ − μ₋| · d` added at the
  selected layer from the last prompt token onward (and every generated token),
  `d` = unit diff-of-means (`dm`) or probe direction, random unit control;
  greedy decoding (the benchmark's deterministic setting); evaluation rows =
  `steer.n_per_class` positives and negatives from the test problems, nested
  across sizes (`pick_eval_rows`). The first parametrisation (class gap *along
  the probe axis*, a fraction of a percent of the residual norm) did nothing
  and was abandoned; records carry `unit_kind = "dm_norm"`.
- Verdicts: `run/prove.py` — the benchmark's prover prompt, DeepSeek-Prover-V2-7B
  via vLLM (2 samples, T 0.6, top-p 0.95, 4096 tokens) for the statement and
  its full negation (`negate_full`), candidates kept only if they restate the
  signature; Lean + Mathlib compile (`analysis/lean.py::check_decls`, direct
  toolchain binary with `LEAN_PATH`, never `lake env`); verdict TC_Fail /
  Proved / Refuted / Unknown.
- Test: `prove.py::dose_response` — slope of the outcome rate on α (α = 0
  included) with a permutation p-value that shuffles α *within each step*
  (2000 permutations), plus the random family's rates.

### 8.2 Residual-stream steering (dm direction)

""")


def trend_rows(folder, tasks=("proved", "sycophancy"), dir_filter=None, outcome_filter=("Proved", "Refuted", "TC_Fail")):
    p = J(folder, "results.json")
    if not p:
        return [], None
    rows = []
    for t in p["trends"]:
        if t["task"] not in tasks or t["outcome"] not in outcome_filter:
            continue
        if dir_filter and not dir_filter(t.get("direction", "dm")):
            continue
        rb, rr = t["rate_by_alpha"], t["random_rate"]
        lo, hi = min(rb, key=float), max(rb, key=float)
        star = "**" if t["perm_p"] < 0.01 else ""
        rows.append([t["model"], t["task"], "true" if t["label"] == 1 else "neg", t.get("direction", "dm"), t["outcome"], t["n_steps"],
                     f"{f(rb[lo])} ({float(lo):+g}) → {f(rb['0.0'])} → {f(rb[hi])} ({float(hi):+g})", f"{star}{t['range_effect']:+.3f} (p={t['perm_p']:.3f}){star}",
                     f"{f(rr.get(min(rr, key=float)))} / {f(rr.get(max(rr, key=float)))}" if rr else "–"])
    return rows, p


rows, p = trend_rows("prove", dir_filter=lambda d: d == "dm")
hdr = ["model", "task", "steps", "direction", "outcome", "n", "rate at min α → 0 → max α", "effect over range (perm. p)", "random (min α / max α)"]
P(f"{p['n_statements']} unique statements proved; verdict counts {p['verdicts_overall']}.\n\n" + tbl(hdr, rows))
P(verify([("files", "`prove/results.json` (`table` per cell, `trends`), `prove/verified.json`, `<model>/steer/<task>/generations.jsonl`"),
          ("code", "`run/steer.py`, `run/prove.py` (`run_prove_gen`, `run_prove_verify`, `dose_response`), `analysis/lean.py`"),
          ("recompute", "`steer` (GPU) → `prove-gen` (GPU, bmuq env, `prove.shard=k/n`) → `prove-verify` (CPU)"),
          ("note", "n ≈ 300 cells are the 300-per-class rerun (alphas ±4, ±8); n ≈ 150 cells keep the ±2 grid")]))

P("### 8.3 Attention-head steering (top-16 heads, α in units of each head's class-mean distance)\n\n")
rows, p = trend_rows("prove_heads", dir_filter=lambda d: d == "dm")
if p:
    P(f"{p['n_statements']} unique statements.\n\n" + tbl(hdr, rows))
P(verify([("files", "`prove_heads/results.json`, `<model>/steer_heads/<task>/{generations.jsonl,steer_heads_k16.npz}`"),
          ("code", "`run/steer.py::run_steer_heads`, `head_class_stats`; `backend.py` `head_steer` hook"), ("recompute", "`steer-heads` → `prove-heads-gen` → `prove-heads-verify`")]))

P("### 8.4 Steering variants (`run/variants.py`; 150 steps per class)\n\n")
for v in cfg["variants"]:
    rows, p = trend_rows(f"prove_{v}", outcome_filter=("Proved",))
    if p:
        P(f"**`{v}`** ({p['n_statements']} statements; Proved outcome only, all directions):\n\n" + tbl(hdr, rows))
P(verify([("files", "`prove_<variant>/results.json`, `<model>/steer_<variant>/<task>/generations.jsonl`"),
          ("code", "`run/variants.py::run_steer_variant`; variant specs in `configs/default.yaml::variants`"),
          ("recompute", "`steer-variant --variant <v>` → `prove-variant-gen --variant <v>` → `prove-variant-verify --variant <v>`")]))

# UFLB summary across sets
P("### 8.5 One-number summary: does any intervention lower UFLB on the sycophancy rows?\n\n")
from leansteer.analysis.metrics import uflb_from_rates  # noqa: E402


def uflb_cells(c1, c0):
    """UFLB from the per-cell rates of the true (c1) and corrupted (c0) evaluation rows."""
    return uflb_from_rates(c1["refuted_rate"], c0["proved_rate"], c1["tc_fail_rate"] + c1["af_fail_rate"], c0["tc_fail_rate"] + c0["af_fail_rate"])


sets = {"residual (steer)": "prove", "heads top-16": "prove_heads"} | {v: f"prove_{v}" for v in cfg["variants"]}
rows = []
for name, folder in sets.items():
    p = J(folder, "results.json")
    if not p:
        continue
    tab = [x for x in p["table"] if x["task"] == "sycophancy" and x["role"] == "eval"]
    cells = []
    for m in MODELS:
        base = {x["label"]: x for x in tab if x["model"] == m and x["alpha"] == 0.0}
        if 1 not in base or 0 not in base:
            cells.append("–"); continue
        b_u = uflb_cells(base[1], base[0])
        best = {"dir": 9, "rnd": 9}
        for d, a in sorted({(x["direction"], x["alpha"]) for x in tab if x["model"] == m and x["alpha"] != 0.0}):
            c = {x["label"]: x for x in tab if x["model"] == m and x["direction"] == d and x["alpha"] == a}
            if 1 not in c or 0 not in c or c[0]["n"] < 100:
                continue
            delta = uflb_cells(c[1], c[0]) - b_u
            key = "rnd" if ("random" in d) else "dir"
            best[key] = min(best[key], delta)
        cells.append(f"{best['dir']:+.3f} / {best['rnd']:+.3f}" if best["dir"] < 9 else "–")
    rows.append([name] + cells)
P(tbl(["intervention family", *MODELS], rows))
P("Cells: largest UFLB *decrease* (ΔUFLB, negative = more faithful) over the family's non-random conditions / the same for its random control (cells with ≥ 100 steps; UFLB from the per-cell Proved/Refuted/TC_Fail/AF_Fail rates in `results.json::table`).\n\n")
P(verify([("files", "`prove*/results.json::table`"), ("code", "this script, `build_background.py` (same arithmetic as the README section)")]))
P("""_Interpretation (human):_ no family lowers UFLB beyond its random control at a
matched magnitude. The thinking models' best decreases are ≤ 0.03 except
StepFun's layer sweep (−0.057), where random does better (−0.074); Herald's
head steering shows −0.075 at α = −8, but the head family has random controls
only at ±32, where steering is far worse than random (outputs collapse), and
the −8 gain is mostly fewer type-check failures. Numbers that look
anti-sycophantic at a single layer come with an equal drop on true steps.
Multi-layer steering at α ±4 destroys the thinking models' outputs where random
directions do not; ablation leaves sycophancy and Kimina's provability
untouched (redundant representation); Kimina's provability effect survives
generation-only steering and lives at the last layer.

""")

# ----------------------------------------------------------------------------- 9 relabel
P("""## 9. Robustness to the prover (Goedel-Prover-V2-8B relabel)

**TL;DR.** 1,500 held-out benchmark statements per model re-proved with a
different prover: verdict agreement 0.72–0.78 (0.88–0.93 on proved-vs-not);
probe AUROC against the new labels within ≈ 0.03 of the benchmark labels on
the same rows for every model and task.

""")
rl = J("relabel_goedel", "results.json")
if rl:
    rows = []
    for m, res in rl["models"].items():
        for task, x in res["tasks"].items():
            rows.append([m, task, f"{res['n']} / agree {f(res['agreement'])} ({f(res['proved_agreement'])} proved-vs-not)", x["layer"], f"{x['n_bench']} / {x['n_new']}",
                         f(x.get("probe_auc_bench_labels"), 3), f(x.get("probe_auc_new_labels"), 3), f(x.get("cv_auc_bench_labels"), 3), f(x.get("cv_auc_new_labels"), 3), f"{f(x.get('probe_auc_agreed_rows'), 3)} ({x.get('n_agreed', '–')})"])
    P(tbl(["model", "task", "rows / verdict agreement", "layer", "n bench / new", "fixed dir: bench labels", "fixed dir: Goedel-Prover labels", "CV: bench", "CV: new", "fixed dir on agreed rows (n)"], rows))
P(verify([("files", "`relabel_goedel/{rows.json,statements.json,candidates.jsonl,verified.json,results.json}`"),
          ("code", "`analysis/relabel.py` (`benchmark_statement`, `run_relabel_select`, `run_relabel_eval`), `run/prove.py::unnegate_full`"),
          ("recompute", "`relabel-select` → `relabel-gen` (GPU, bmuq) → `relabel-verify` → `relabel-eval`"),
          ("note", "fixed-direction AUROCs are higher than on the full test set because this subset is uniformly sampled, not class-balanced; compare bench vs new on identical rows")]))

# ----------------------------------------------------------------------------- 10 external
P("""## 10. Generalization beyond FaithformBench

**TL;DR.** Four external sets through the same pipeline (activations,
autoformalization with the four models, our prover pass, transfer
evaluation). On raw PRM800K (human ratings) and miniF2F (488 statements, LLM
corruptions) the FaithformBench truth direction transferred weakly and
retrained probes barely beat text. Restricting PRM800K to steps whose rating
agrees with an LLM judge's claim verdict lifts retrained truth probes to
0.67–0.77 vs text 0.63 and provability probes to 0.76–0.85 vs 0.58–0.64;
enlarging miniF2F to 1,988 statements makes the corruption readable at
0.67–0.71 vs text 0.64. The representation generalizes; the direction must be
refitted; label quality and statement granularity set the gap over text.

""")
for name, title in (("prm800k", "PRM800K phase-2 test, raw human ratings"), ("prm800k_claims", "PRM800K restricted to claim-level steps (LLM judge)"),
                    ("minif2f", "miniF2F (488) + adversary-prompt corruptions"), ("minif2f_ext", "miniF2F + 1,500 Lean-Workbook statements, corrupted")):
    e = J(f"ext_{name}", "eval.json"); p = J(f"ext_{name}", "prove", "results.json")
    if not e:
        continue
    P(f"### 10.{list(('prm800k', 'prm800k_claims', 'minif2f', 'minif2f_ext')).index(name) + 1} {title} — {e['n']} rows, {e['n_true']} true\n\n")
    if p:
        P("Benchmark metrics of the four autoformalizers on this data (our prover pass; UFLB = ½FNR + ½FPR + AFFR, lower is better):\n\n" + tbl(["model", "TPR", "FNR", "FPR", "AFFR", "UFLB", "true: Proved / Refuted / Unknown / TC_Fail", "false: Proved / Refuted / TC_Fail"],
            [[r["model"], f(r["TPR"]), f(r.get("FNR")), f(r["FPR"]), f(r.get("AFFR")), f"**{f(r.get('UFLB'))}**", f"{f(r['true_proved'])} / {f(r['true_refuted'])} / {f(r['true_unknown'])} / {f(r['true_tc_fail'])}", f"{f(r['false_proved'])} / {f(r['false_refuted'])} / {f(r['false_tc_fail'])}"] for r in p["table"]]))
    rows = []
    for m, r in e["models"].items():
        t = r["truth"]
        def g(prefix):
            return next((v for k, v in t.items() if k.startswith(prefix)), None)
        bl = t.get("best_layer_cv", {})
        rows.append([m, f(g("perturbed_detect_dir"), 3), f(g("sycophancy_dir"), 3), f(g("proved_strict_dir"), 3), f(g("retrained_cv"), 3), f"{f(bl.get('auc'), 3)} (L{bl.get('layer', '?')})", f(t.get("tfidf_cv"), 3)])
    P("Truth (true vs false step) from the prompt_last state:\n\n" + tbl(["model", "FB corruption-detector dir", "FB sycophancy dir", "FB proved_strict dir", "retrained CV (FB layer)", "best-layer CV (optimistic)", "TF-IDF CV"], rows))
    rows = []
    for m, r in e["models"].items():
        for task, x in r["verdict_tasks"].items():
            rows.append([m, task, f"{x['n']} ({x['pos']})", f(x.get("bench_dir_auc"), 3), f(x.get("retrained_cv_auc"), 3), f(x.get("tfidf_cv"), 3)])
    P("Verdict tasks on our prover's verdicts:\n\n" + tbl(["model", "task", "n (pos)", "FB direction", "retrained CV", "TF-IDF CV"], rows))
P(verify([("files", "`ext_<name>/{steps.parquet,eval.json,verdicts.json}`, `ext_<name>/<model>/{acts/,generations.jsonl}`, `ext_<name>/prove/results.json`"),
          ("inputs", f"`{lcfg.SCRATCH_ROOT}/external/prm800k/{{phase2_test.jsonl,judged.jsonl}}`, `external/minif2f/perturbed.jsonl`, `external/minif2f_ext/{{statements.jsonl,perturbed.jsonl}}`"),
          ("code", "`data/external.py` (loaders, filters, balanced problem-grouped sampling), `run/external.py`, `analysis/external_eval.py`; judge `scripts/judge_prm800k.py`; corruption `scripts/perturb_minif2f.py` (the benchmark's adversary prompt, Gemma-3-12B-it)"),
          ("judge statistics", "of 5,067 steps rated −1: 1,687 FALSE_CLAIM, 3,310 TRUE_CLAIM, 70 NO_CLAIM; of 5,000 rated +1: 4,494 TRUE_CLAIM, 337 FALSE_CLAIM, 169 NO_CLAIM (gpuq log of `scripts/ext2_jobs.sh prm`)"),
          ("recompute", "`scripts/external_job.sh <name> <gpu_util>` runs ext-collect → ext-generate → ext-eval → ext-prove-gen → ext-prove-verify → ext-eval")]))

# ----------------------------------------------------------------------------- 11 conclusions
P("""## 11. Conclusions, caveats, open ends

**What the evidence supports (human synthesis).**

1. Before writing any Lean, the four autoformalizers linearly encode whether a
   step is true (corruption detector 0.69–0.80 vs text 0.54) and, among true
   steps, whether it is provable (0.66–0.75). The signal is sparse in a few
   late heads, stable to prover choice, not a text/difficulty artefact, and
   has ±0.02 intervals (§4–6, §9).
2. The `sycophancy` and `proved_strict` probes mostly inherit the truth
   signal; sycophantic *repair* among corrupted steps is only weakly readable
   (0.56–0.69 ≈ text) (§5.2).
3. The probe is useful as an abstention signal: −20 % coverage lowers UFLB
   (see §7 for the exact values) and removes 40–54 % of sycophantic cases.
4. Causally, the direction moves provability for Kimina (and formalization
   decisiveness for Goedel) and nothing else; sycophancy is not steerable at
   either site, any layer, or by ablation, with MDE ≈ 0.11 (§8).
5. The representation generalizes to clean claim-level labels and to
   competition statements with refitting; the specific direction is
   benchmark-shaped (§10).

**Caveats to state in the report.** The af_v1 prompts carried the bare step,
not the benchmark's `<context>` block (§12: for Goedel this understates the
corruption-detecting probes by 0.05–0.12 AUROC and leaves the provability,
repair and steering conclusions unchanged); Lean type-check proxy uses Mathlib v4.33.1
(not the benchmark's); prover runs use 2 samples (DeepSeek) / 1 (Goedel-Prover);
teacher-forced probes exist for true steps only; the PRM800K judge is a model
(7 % of +1 steps judged false claims); Lean-Workbook corruptions are more
text-detectable than miniF2F's; steering interventions are additive or
projective along probe/mean-difference directions — learned or non-linear
interventions were not tried.

**Infrastructure pitfalls that affected results and were fixed:** Herald and
DeepSeek-Prover tokenizers under transformers ≥ 5 (fast-tokenizer copy +
round-trip assert); the first steering step size; probe summaries overwritten
by partial reruns (now merged); `lake env` in a shared Mathlib checkout tried
to re-clone it (replaced by a direct `lean` call).
""")

P("""## 12. Context prompts: Goedel re-run with the benchmark's `<context>` block (run `af_ctx`)

**TL;DR.** The af_v1 prompts omitted the parent steps the benchmark put in
every prompt. With them, Goedel's pre-generation probes gain 0.05–0.12 AUROC
on every task whose negatives are corrupted steps (sycophancy 0.78 → 0.86,
proved_strict 0.78 → 0.86, corruption detector 0.75 → 0.83; text baselines
move ≤ 0.05), the proved direction within true steps is unchanged (0.72), the
sycophancy direction still cannot tell repaired from refuted corrupted steps
(0.39), the abstention gate does not improve (UFLB 0.177 vs 0.166), and the
steering effects shrink (corrupted-step refutation range +0.12 → +0.05). The
context tells the model *that* a step is wrong, not what it will do about it.

- Where: `context_comparison.md` (repo root; every table side by side, with a
  findings section) generated by `scripts/compare_context.py` from
  `$LEANSTEER_SCRATCH/runs/af_ctx/goedel/{probes,probes_tf,controls,uncertainty,abstain,baselines,steer}` and `runs/af_ctx/prove`.
- How it ran: `scripts/context_job.sh` (collect → collect-tf → probe → probe-tf
  → steer, on a shared-card slice), `scripts/context_cpu_job.sh` (controls,
  uncertainty, abstain, transfer, baselines), `scripts/context_steer_resume.sh`
  (the sycophancy steering was resumed after an out-of-memory kill),
  `scripts/context_prove_job.sh` (DeepSeek-Prover, whole card). Settings as
  af_v1's 300-per-class residual steering: dm ±4/±8, random ±8.
- Verify the prompt text: `python -c "from leansteer.data.faithform import load_af, unique_steps; print(unique_steps(load_af(models=['goedel'])).informal.iloc[5])"`
  against `veriform/preprocessing/dag.py::DAGNode.contextualized` in the
  benchmark supplement.
- Audit checks 1–2 on this run: `AUDIT_CONFIG=configs/context.yaml python scripts/audit_qualitative.py`
  → `$LEANSTEER_SCRATCH/audit/af_ctx/audit_checks12.json` (true-vs-corrupted
  0.70 → 0.79, true rows refused 8 % → 3 %, within-corrupted 0.33 → 0.36).
- Not re-run with context: Kimina, Herald, StepFun; attention heads; steering
  variants; external datasets. Their af_v1 numbers stand as bare-step results.
""")

sys.stdout.write("\n".join(out))
