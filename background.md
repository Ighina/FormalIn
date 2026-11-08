# LeanSteer — background

What this repository does, why, how, and what came out of it. Each macro-section
opens with a TL;DR; the rest of the section goes into the details and points at
the code that implements them. Result tables live in `README.md` and in the
generated report (`$LEANSTEER_SCRATCH/runs/af_v1/report/summary.md`); this
document explains them.

Code paths below are relative to the repository root. `leansteer/` is about
2,900 lines of Python; the stages are driven by `python -m leansteer.cli <stage>`.

---

## 1. The question

**TL;DR.** FaithformBench scores four Lean autoformalizers by what a prover can
do with their output. We ask whether that prover verdict is already encoded in
the autoformalizer's residual stream *while it reads the step*, before it writes
any Lean; whether the direction that encodes it is causally usable; and whether
either finding is an artefact of text, dataset or difficulty.

An autoformalizer turns a natural-language mathematical step into a Lean
statement. FaithformBench ([`../RLLF/rllf/data/faithform.py`](../RLLF/rllf/data/faithform.py)
has the companion loader) takes ProcessBench reasoning traces, splits them into
atomic steps, corrupts each step with a "BrokenMath-style" LLM perturbation,
autoformalizes both the true and the corrupted step with four dedicated models
(Goedel-Formalizer-V2-8B, Kimina-Autoformalizer-7B, Herald_translator,
StepFun-Formalizer-7B), and runs DeepSeek-Prover-V2 on the statement and on its
negation. The verdict per (step variant, model) is

| verdict | meaning |
|---|---|
| `Proved` | the prover proved the formal statement |
| `Refuted` | the prover proved its negation |
| `Unknown` | neither |
| `TC-Fail` | the statement does not elaborate (type-check) |
| `AF-Fail` | the autoformalizer produced no theorem |
| `Declarative` | the step is not a proposition; never autoformalized |

A *faithful* autoformalizer should get true steps `Proved` and corrupted steps
`Refuted`; a corrupted step that ends up `Proved` is the benchmark's notion of
**sycophancy** (the formalizer silently repaired the error).

The three questions this codebase answers, in order:

1. **Probing.** Can a linear probe on the residual stream, read before
   generation, predict the verdict? For which verdict distinctions, at which
   layers, and how far above text-only baselines?
2. **Confounds.** Is the probe reading the step's truth/provability, or a
   proxy for length, dataset, step type or problem difficulty?
3. **Steering.** Does pushing the residual stream along the probe direction
   change the formalization in the direction the probe predicts, as judged by
   Lean and by the prover? And do the probe results survive swapping the prover?

---

## 2. Input data and how it is processed

**TL;DR.** We read `Faithform-AF.csv` (one row per step variant × model,
111,464 rows), normalise the verdicts, drop the 9,192 Declarative rows that were
never autoformalized and 30 corrupted steps whose text equals the true step,
and build four binary tasks per model with problem-level train/test splits. The
file keeps the autoformalizer's raw response only for true steps; for corrupted
steps it stores prover artefacts instead. That asymmetry is why the main probes
read the model *before* it generates.

### 2.1 The CSV

Columns we use (`leansteer/data/faithform.py::load_af`):

- `problem_id` (e.g. `math-548`, `gsm8k-250`; the prefix is the benchmark
  subset: gsm8k, math, olympiadbench, omnimath), `node_id` (`step_k`),
  `is_perturbed`, `step_content`, `autoformalizer_model`, `proof_result`,
  and — when `with_outputs=True` — `autoformalizer_output` and `prover_output`.
- Every `(problem_id, node_id, is_perturbed)` triple appears once per model:
  15,082 true steps and 12,784 corrupted ones; every corrupted node has a true
  counterpart with the same `(problem_id, node_id)`.

Two facts about the file shaped the design (both verified in
[`faithform.py`](leansteer/data/faithform.py)'s module docstring):

- **Declarative rows have no model output**: the benchmark skipped them, so
  they carry no verdict about the model and are dropped (`drop_declarative`).
- **`autoformalizer_output` is not one column.** For true steps it is the raw
  response (Goedel/StepFun emit a `<think>…</think>` chain of thought before
  the Lean block). For corrupted steps it holds whatever the prover side
  produced: the proof (`Proved`), the negation proof `theorem not_…_full : ¬ (∀ …)`
  (`Refuted`), the statement with `sorry` (`TC-Fail`/`Unknown`) or
  `-- Failed to formalize` (`AF-Fail`); `prover_output` is empty there.
  Consequently the model's own response is available for only half the rows.

`extract_statement` recovers a `theorem … := by sorry` statement from either
kind of text (last fenced theorem ending in `sorry`, or the signature of a
proof up to `:= by`); negation proofs are un-negated later by
`leansteer/run/prove.py::unnegate_full` when the benchmark's statements are
needed (§8).

### 2.2 Tasks

`faithform.py::TASKS` defines positives and negatives as selectors over a
model's rows; `build_task` materialises a table with `label` and
`group = problem_id`:

| task | positives | negatives |
|---|---|---|
| `proved` | true step, Proved | everything else; negatives subsampled to the positive count (`balance`) |
| `proved_strict` | true step, Proved | corrupted step, Refuted |
| `failing` | verdict ∉ {AF-Fail, TC-Fail} | AF-Fail or TC-Fail |
| `sycophancy` | true step, Proved | corrupted step, Proved |
| `proved_unp`, `failing_unp` | the same restricted to true steps (for the teacher-forced anchors, §4.4) |

`Unknown` counts as *not failing* (the statement elaborated; the prover was
undecided). Sizes per model are printed by `python -m leansteer.cli data`:
roughly 10–19k rows for `proved`, 9–16k for `proved_strict`, 25.5k for
`failing`, 7–12k for `sycophancy`.

### 2.3 Splits

`faithform.py::problem_split` assigns whole problems to train or test
(`analysis.test_frac = 0.3`, seeded), so a corrupted step never sits on the
other side from its true counterpart and nothing about a problem leaks across
the split. A second problem-level split inside train (`val_frac = 0.2`) is
used only to select the layer.

### 2.4 Metadata used by the baselines

`FaithformBench.csv` (one row per step variant, no model dimension) adds
`statement_type` (computation, logical, substitution, general_knowledge,
simplification, verification) and `proof_type` (deduction, assumption, axiom);
`Faithform-LLM.csv` holds the verdicts of four frontier-LLM formalizers
(gpt-5.2, gemini-3.1-pro, claude-opus-4-7, qwen-plus) on the same steps, joined
on `(problem_id, step_content)` because that file renumbered its nodes. Both
are read in `leansteer/analysis/baselines.py` (§6).

---

## 3. The models and what we read from them

**TL;DR.** The four autoformalizers are run with the benchmark's verbatim
prompts. A hooked Hugging Face backend captures the residual stream after every
decoder block at chosen token positions ("anchors") in a single prefill, and
can add or project out a vector at a layer during generation. Prompt-side
anchors exist for every row; teacher-forced anchors (the model's own recorded
output appended) exist for true steps only.

### 3.1 Prompts — `leansteer/models/registry.py`

`ModelSpec` records, per model, the Hugging Face id, the system prompt, the
user template and two quirks copied from the benchmark's
`veriform/formalization/fine_tuned.py`: Goedel's user prompt has *no separator*
between the step and "Think before you provide the lean statement." (its
training format), and StepFun's prompt gets a literal `<think>` appended after
the chat template to switch its R1-distilled base into reasoning mode. Chat
formatting uses each repository's own `chat_template`; `build_prompt` is the
single place that turns a step into the exact string the benchmark fed the
model, so the activations we read are those of the benchmark's forward pass.

Herald is a Llama-2 derivative whose tokenizer, under transformers ≥ 5,
resolves to the slow sentencepiece class and silently drops every space.
`leansteer/models/backend.py::load_tokenizer` forces the `tokenizer.json`
backend for it (`fast_tokenizer=True`) and asserts that a prompt decodes back
to itself; the first Herald run, made before this guard existed, was discarded.

### 3.2 Hooked backend — `leansteer/models/backend.py`

`HFBackend` loads the model in bfloat16 on one GPU and registers forward hooks
on the embedding (layer 0) and on each decoder block (layer `l` = residual
stream *after* block `l`). One hook does three jobs:

- **capture**: given per-row weight vectors over prompt positions
  (`anchor_weights`), it returns `einsum('abt,btd->abd')` of the weights and
  the hidden states, so arbitrary anchors — last token, mean over a span — are
  read in one pass with left padding handled by the weights;
- **steer**: add a vector at the anchor position and every later position
  during prefill, and at every generated position during decoding
  (`Intervention(kind="steer")`);
- **project**: set the component along a unit direction to a target value
  (`kind="project"`, not used in the reported runs).

`capture` runs a prefill with `logits_to_keep=1` and also records next-token
entropy and top probability at the last prompt token (nuisance covariates
later). `generate` is greedy decoding — the benchmark's `deterministic`
setting — with the steering hooks left in place.

### 3.3 Anchors and the collection stage — `leansteer/run/collect.py`

`run_collect` builds the prompt for every step variant of a model (25,538 per
model after the drops of §2), finds the token span of the step text inside the
prompt via offset mappings (`backend.step_span`), and stores float16 memmaps
`acts/<anchor>.npy` of shape `[N, n_layers+1, d]` plus `index.json` (row order)
and `meta.jsonl` (prompt length, entropy, top token). Anchors:

- `prompt_last` — last prompt token, the state from which the model starts
  writing its formalization;
- `step_last` — last token of the step text inside the prompt;
- `step_mean` — mean over the step's tokens.

Rows are processed longest-first for batching and the stage is resumable by
row count (`progress.json`). Model sizes: Goedel 36 layers × 4096, Kimina and
StepFun 28 × 3584, Herald 30 × 4096; about 23 GB of activations per model.

`run_collect_tf` (§4.4) does the same for the *teacher-forced* anchors on true
steps: the recorded output is appended to the prompt (stripping a leading
`<think>` when the prompt already ends with it) and the residual stream is read
at `output_last`, `statement_last` and `statement_mean` (span of the extracted
theorem inside the output; outputs over 3,072 tokens are skipped). Store:
`acts_tf/`.

---

## 4. Probing

**TL;DR.** Per model, anchor and task, a standardised logistic-regression
probe is fitted at every other layer on the training problems, the layer is
chosen on a validation split of train, and AUROC is reported on held-out
problems. Pre-generation probes reach 0.66–0.80 for `proved`, `proved_strict`
and `sycophancy` against TF-IDF text baselines of 0.54–0.62, peak in the last
third of the network, and share one direction across those three tasks.
`failing` sits at the text baseline. After the model has written its output
(teacher-forced anchors) the verdict is far more readable, 0.82–0.96.

### 4.1 Protocol — `leansteer/analysis/probes.py`

`run_probes` → for each task table (§2.2) restricted to rows with activations:

1. problem-level train/test split; inner problem-level validation split;
2. `fit_layer` at each layer in `analysis.layer_stride` steps (plus the last):
   standardise, `LogisticRegression(C=0.05, class_weight="balanced")`,
   validation AUROC for selection, test AUROC / average precision / balanced
   accuracy; also the diff-of-means direction's AUROC;
3. the selected layer is the one with the best *validation* AUROC (`selected`
   in `result.json`); `best_test_auc` is reported separately and never used
   for selection;
4. per-layer directions are saved (`direction_L<k>.npz`: LR direction in raw
   activation space, diff-of-means, class-mean projections) for steering;
5. text baselines (`text_baselines`): TF-IDF (1–2-grams) logistic regression
   on the step text, and a length-only model; for the teacher-forced store a
   TF-IDF model that also sees the recorded output;
6. `cross_task_transfer`: the selected direction of task A scored on task B's
   test rows.

The report stage (`leansteer/analysis/report.py`) draws AUROC-by-layer curves
(`report/auroc_<model>_<anchor>.png`).

### 4.2 What the pre-generation probes show

Test AUROC at the selected layer (best of `prompt_last` / `step_mean`); the
TF-IDF step-text baseline in parentheses:

| model | proved | proved_strict | failing | sycophancy |
|---|---|---|---|---|
| goedel | 0.75 (0.59) | 0.80 (0.55) | 0.67 (0.68) | 0.78 (0.62) |
| stepfun | 0.71 (0.59) | 0.78 (0.55) | 0.74 (0.70) | 0.77 (0.60) |
| kimina | 0.66 (0.59) | 0.73 (0.54) | 0.68 (0.68) | 0.75 (0.57) |
| herald | 0.69 (0.60) | 0.79 (0.56) | 0.67 (0.64) | 0.77 (0.58) |

- AUROC rises monotonically with depth and plateaus in the last third
  (selected layers 22–36 of 36 for Goedel, 28 of 28 for Kimina/StepFun).
- The `proved`, `proved_strict` and `sycophancy` directions transfer to each
  other within about 0.03 of the native probe (e.g. Goedel: proved→sycophancy
  0.77 vs 0.78 native), while `failing` neither transfers nor receives
  transfer. Reading: there is one "this step is true and provable" direction;
  the three tasks are different slices of it.
- `failing` is at or below the text baseline for every model: whether the
  formalization will fail to parse or elaborate is not readable from the
  pre-generation state (and §6 shows it is mostly a problem-level property).

### 4.3 Why pre-generation anchors

Because the raw response exists only for true steps (§2.1), any anchor placed
on the model's output could be computed for half the data at most, and never
for the corrupted steps that `proved_strict` and `sycophancy` need. The
pre-generation anchors treat all rows alike and ask the sharper question: does
the model *already know*, from reading the step, how the verdict will go.

### 4.4 Teacher-forced anchors — `run_collect_tf`, `probe-tf`

On true steps we also append the model's recorded output and probe the
post-output state for `proved_unp` and `failing_unp`. The fair text baseline
here must see the Lean output too (`tfidf_output_auc`). Results (statement_mean
anchor): proved 0.82–0.86, failing 0.91–0.96. The output-aware TF-IDF model
matches Goedel's post-output probe (0.82 vs 0.86) but not Kimina's, Herald's
or StepFun's (0.73–0.76 vs 0.82–0.86 on proved). Conclusion: once the
formalization is written, the verdict is largely determined and largely
visible in the text; the interesting signal is the earlier one.

### 4.5 Attention-head probes — `leansteer/analysis/heads.py`

Following Inference-Time Intervention, `run_collect_heads` stores every head's
output (the input of each block's `o_proj`, `n_heads × 128`) at `prompt_last`
and `step_mean` (`backend.capture(component="heads")`, a pre-hook on `o_proj`),
and `run_head_probes` fits, on the same splits as the residual probes, a probe
per head (layer × head heat-map), a probe on all heads of a layer, and a probe
on the top-k heads by validation AUROC. Findings (prompt_last): a single
well-chosen head matches the residual probe (Goedel `proved` L36H8 0.76 vs
0.72; `sycophancy` L23H15 0.79 vs 0.78); the signal is sparse (median head
near chance, a few dozen to a few hundred heads above 0.6); and concatenating
4–64 heads beats the residual probe by 0.05–0.11 on the three verdict tasks
for every model (Goedel sycophancy 0.87 vs 0.78, Herald proved_strict 0.81 vs
0.71, Kimina sycophancy 0.80 vs 0.69). The informative heads sit where the
residual probes peak and recur across the verdict tasks; `failing` heads sit
mid-network and stay at the text/difficulty baselines. Reading the attention
outputs before they are summed into the residual stream and mixed with the
MLP path recovers information a single residual direction blurs.

---

## 5. Steering: generation-level readouts

**TL;DR.** We add `α · |μ₊ − μ₋| · d` to the residual stream at the selected
layer from the last prompt token onward, with `d` the diff-of-means or the
probe direction and a random direction as control, and regenerate greedily.
Without a prover we measure parse rate, statement change, and Lean
type-checking of the generated statement. The step size matters: the first
attempt used the class gap *along the probe axis*, a fraction of a percent of
the residual norm, and did nothing.

### 5.1 Parametrisation — `leansteer/run/steer.py`

`run_steer` loads the task's selected layer and direction, computes the class
means on the training rows at that layer (`class_stats`), and sets
`unit = ‖μ₊ − μ₋‖`. Directions: `dm` (unit diff-of-means), `probe` (LR
direction, sign fixed so +α points to the positive class) and `random` (unit
Gaussian). `α ∈ {−8, −4, −2, 2, 4, 8}` for `dm`/`probe`, `±8` for `random`,
plus the unsteered α = 0 generation. Probe directions are nearly orthogonal to
the mean difference (cos ≈ 0.03–0.11) and the class gap along them is 1–7 in
units where `‖μ₊ − μ₋‖` is 13–130, which is why the gap along the probe is not
a usable unit (`UNIT_KIND = "dm_norm"` tags every generation so rows from the
abandoned parametrisation are ignored).

Evaluation rows (`pick_eval_rows`): `steer.n_per_class` positives and negatives
from the test problems; for `sycophancy` the true counterparts of the corrupted
negatives are also generated unsteered as references. Every generation is
written to `steer/<task>/generations.jsonl`; the stage resumes by
`(direction, α, uid)`. Scoring (`score_task`) restricts itself to the rows the
*current* config selects (`current_selection`), so changing `n_per_class` does
not mix selections.

### 5.2 Metrics without a prover

- `parse_rate`: a `theorem … sorry` block was produced (AF-Fail proxy);
- `changed_rate` / `sim_base`: the extracted statement differs from the
  unsteered one / difflib similarity to it;
- `typecheck_rate`: the statement elaborates against Mathlib
  (`leansteer/analysis/lean.py::typecheck`, TC-Fail proxy — same kernel,
  Mathlib v4.33.1 rather than the benchmark's version);
- for `sycophancy`, `match_true_rate` / `sim_true`: does the corrupted step's
  statement coincide with the model's statement for the true step.

### 5.3 What the generation-level readouts show

Kimina gives the cleanest picture: pushing toward "not proved" lowers the
type-check rate dose-dependently (0.88 → 0.60 on `proved`, 0.94 → 0.77 on
`proved_strict` at α = −8), pushing toward "proved" keeps it at or above
baseline, and the random direction changes statements just as often without
moving the type-check rate. For the thinking models almost any intervention
changes the statement (a small perturbation gets amplified through 1–2k tokens
of chain of thought), so `changed_rate` is uninformative and the type-check
proxy is nearly saturated. `match_true_rate` is low even unsteered (≤ 8 %):
the benchmark's sycophantic cases are not literal copies of the true-step
formalization, so sycophancy cannot be judged textually. Both observations
motivated the prover stage.

---

## 6. Baselines and confounds

**TL;DR.** Three covariate groups separate three readings of the probe:
text nuisance, problem-level difficulty, and an *oracle* made of the other
seven formalizers' verdicts on the same step. For `proved`, `proved_strict` and
`sycophancy` the nuisance and difficulty covariates reach only 0.54–0.72,
explain ≤ 13 % of the probe score, and the probe keeps 0.65–0.76 after they
are regressed out; its AUROC is flat across benchmark subsets and step types
and survives inside strata of the oracle. `failing` is the opposite: a
problem-level property the probe adds nothing to. A fine-tuned ModernBERT on
the step text is below every probe on the three verdict tasks and above the
probe on `failing`.

### 6.1 Covariates — `leansteer/analysis/baselines.py::build_features`

- **nuisance**: log chars, log words, counts of numbers, operators, `=`,
  fractions, LaTeX, alphabetic words; step position in the trace, relative
  position; next-token entropy, top probability and prompt length at the
  anchor (from `meta.jsonl`); a 50-dimensional TF-IDF-SVD embedding of the
  step fitted on train. This is the REVISE nuisance set adapted to a
  single-input setting.
- **difficulty**: benchmark subset (one-hot), number of steps, leave-one-out
  proved and failing rates of the *other* true steps of the same problem for
  this model (`_loo_problem_rate`) and pooled over all formalizers
  (`_loo_problem_rate_all`), `statement_type`, `proof_type`, length.
- **covariates** = nuisance ∪ difficulty.
- **oracle**: proved / refuted / failing indicators of the other 3 dedicated
  and 4 LLM formalizers on this very step variant, and their counts
  (`other_verdicts`). These encode whether the step is *true and formalizable*
  — the signal itself — so they are reported as a ceiling and used to
  stratify, never residualised as if they were a nuisance. (The first version
  of this analysis mixed them into "difficulty"; regressing them out removed
  the signal, which is exactly what one expects of an oracle, and the groups
  were split before the reported runs.)

### 6.2 Analyses — `run_baselines`

Per model and task at the `prompt_last` anchor and selected layer:

- group-only AUROC (logistic regression on the covariates);
- stacked AUROC (covariates + the probe's out-of-fold score; `_oof_scores`)
  and the gain over the covariates alone;
- residualised probe: ridge-regress the covariates out of the activations and
  refit the probe;
- R² of the probe score on each covariate group;
- within-stratum AUROC on the test rows by subset, by `statement_type`, and by
  `n_others_proved` (`_strata_auc`), plus probes trained *and* tested inside
  one subset (`within_subset_trained`);
- correlations of the probe score with single covariates.

### 6.3 ModernBERT — `leansteer/analysis/modernbert.py`

`answerdotai/ModernBERT-base` is fine-tuned per (model, task) on the same
problem-level split (3 epochs, lr 3e-5, max 160 tokens, class-weighted BCE,
checkpoint chosen on validation AUROC). It sees the step text only.

### 6.4 Conclusions

- For `proved`, `proved_strict`, `sycophancy`: nuisance 0.53–0.63, difficulty
  0.54–0.72, R² of the probe score on the covariates 0.00–0.13, residualised
  probe 0.65–0.76, gain beyond covariates +0.07 to +0.23. AUROC is within a
  few hundredths across gsm8k / math / olympiadbench / omnimath, also when
  trained inside one subset, and across statement types. Within strata of
  `n_others_proved` (0 … 7) it stays 0.55–0.85. The probe score correlates
  with the oracle at r ≈ 0.35–0.40 and with problem-level proved rate at
  r ≈ 0.0–0.2. **The probe is not a difficulty direction; it reads the step's
  truth/provability, with a model-specific residue.**
- For `failing`: difficulty covariates alone reach 0.75–0.86 (the problem's
  failing rate is the strongest feature), the probe adds ≤ 0.01 and collapses
  to 0.52–0.60 after residualisation; ModernBERT on the text (0.71–0.78) beats
  the probe. **Failure is a property of the problem and the step's surface
  form, not something the pre-generation state adds to.**
- ModernBERT on the three verdict tasks: 0.55–0.65, below every probe.

### 6.5 Perturbation-confound controls, uncertainty, abstention — `controls.py`, `uncertainty.py`, `abstain.py`

Three later additions tighten §6. *Uncertainty*: problem-clustered bootstrap
95% intervals (≈ ±0.02) and five split seeds (sd ≤ 0.015) for every headline
AUROC. *Perturbation controls*: because every negative of `sycophancy` and
`proved_strict` is a corrupted step, directions were trained that never see a
perturbation label (`proved_unp`) or only that label (`perturbed_detect`,
a corruption detector) and evaluated inside corrupted steps only (Proved vs
Refuted). The corruption detector reads the state at 0.69–0.80 while the
label is invisible in the text (TF-IDF 0.54); it separates the sycophancy
rows as well as the sycophancy probe (sign flipped) and is anti-aligned with
it (cos −0.28 to −0.38); inside corrupted steps no pre-trained direction
predicts which ones get sycophantically proved (0.39–0.54), and a probe
trained for that question reaches only 0.56–0.69 against text at 0.59–0.64.
The supportable claim is therefore "the model knows the step is false" (and,
among true steps, whether it is provable), not "the model knows it is about
to repair the error". *Abstention*: refusing the 20 % of steps the sycophancy
direction scores lowest raises the benchmark's faithfulness index by
0.03–0.04 for every model and removes 40–54 % of sycophantic cases (random
21 %, text detector 21–24 % with no gain); at 30 % the gain is 0.05–0.06.

---

## 7. Prover-level evaluation of steering

**TL;DR.** The steered generations are judged exactly as the benchmark judges
the originals: DeepSeek-Prover-V2-7B samples proofs for the statement and for
its full negation, Lean checks them, and the verdict follows. With 150 steps
per class and a within-step permutation test, Kimina's `proved` direction is
causally effective (true steps proved 0.37 → 0.61, type-check failures
0.32 → 0.04 from α = −8 to +8; random direction flat). No model shows a
prover-level effect on sycophancy.

### 7.1 Protocol — `leansteer/run/prove.py`, `leansteer/analysis/lean.py`

- `select_generations` collects, for the tasks and conditions in
  `prove.conditions`, every generated statement of the current selection;
  `unique_statements` dedupes by whitespace-normalised text (17,815 statements
  in the final run).
- `negate_full` builds the benchmark's "full" negation
  `theorem not_<name>_full : ¬ (∀ <binders>, <type>) := by sorry` with a
  bracket-aware binder parser; on a sample of real statements, every negation
  elaborated whenever the statement did.
- `prover_prompt` reproduces the benchmark's prompt: the Mathlib header, a
  docstring with the natural-language step, the statement with `sorry`, and
  the "Complete the following Lean 4 code … provide a detailed proof plan"
  wrapper.
- `run_prove_gen` runs vLLM (the `bmuq` env; `prove.shard=k/n` splits the
  pending prompts across cards; resumable by `(statement, kind)`), 2 samples at
  temperature 0.6 / top-p 0.95 / 4,096 tokens, for the statement and for the
  negation. The DeepSeek tokenizer has the same transformers-5 slow-class
  problem as Herald; `prover_tokenizer_dir` hands vLLM a fast-tokenizer copy.
- `extract_candidate` keeps the last Lean block only if it restates the target
  signature verbatim (the benchmark's filter) and contains no `sorry`.
- `run_prove_verify` type-checks statements (cached from the steer stage) and
  compiles candidates in batches of 8 declarations per file with a process
  pool over the toolchain's `lean` binary and an explicit `LEAN_PATH`
  (`lean.py::check_decls`; `lake env` is avoided on purpose — it tried to
  re-clone Mathlib inside a shared project). A declaration passes if no error
  and no `declaration uses sorry` warning falls in its line range. Verdict:
  `TC_Fail` if the statement fails, else `Proved` if any direct candidate
  passes, else `Refuted` if any negation candidate passes, else `Unknown`.
- `score_prove` tabulates verdict rates per (model, task, direction, α, label)
  and `dose_response` fits the slope of each outcome rate on α along the `dm`
  direction, with a p-value from 2,000 permutations of α *within each step*
  (so step difficulty is held fixed) and the random-direction rates for
  contrast.

### 7.2 Results (150 steps per class)

- Unsteered, the regenerated formalizations reproduce the benchmark: true
  steps labelled Proved are re-proved 50–77 % of the time, negatives 10–33 %.
- **Kimina `proved`**: along +α, true steps proved 0.37 → 0.60 → 0.61,
  negatives 0.15 → 0.23 → 0.27, type-check failures 0.32 → 0.05 → 0.04 (all
  p < 0.001); random ±8 gives 0.59/0.56, 0.23/0.21 and 0.05/0.06. This is a
  genuine causal lever on the quantity the probe reads.
- **Goedel `proved`**: proved rates flat, but refutations of negatives rise
  0.26 → 0.40 (p < 0.001): the direction makes Goedel formalize false steps
  more decisively rather than prove them. **Herald**: type-check failures fall
  with +α (p ≈ 0.002), proved flat. **StepFun**: nothing.
- **Sycophancy**: for every model the proved rate of corrupted steps is flat
  under steering (p > 0.3). The direction is readable (§4) but, at this layer
  and magnitude, is not a lever on the behaviour. The earlier 48-per-cell hint
  for StepFun did not survive.

### 7.2b Power: 300 steps per class

The null tasks were rerun with 300 steps per class (nested selection,
unsteered generations reused).  Minimal detectable difference between α = 0
and α = ±8 at 80 % power is ≈ 0.11; the observed sycophancy effects are
−0.05 to +0.04 (all p > 0.09) for every model, and Goedel/StepFun/Herald
`proved` effects are ≤ 0.04.  Kimina's `proved` effect is unchanged.  The
null is therefore "no steering effect on sycophancy larger than about 0.1".

### 7.3 Head-level steering — `run_steer_heads`, `prove-heads-*`

The ITI-style counterpart of §7: the top-16 heads of each task (by
validation AUROC in `probes_heads`) receive `α · |μ₊ − μ₋|_h · d_h` on their
output (a pre-hook on `o_proj`, `Intervention(kind="head_steer")`), α ∈ ±8,
±16, ±32 per head, with random unit directions on the same heads as the
control, on the same 150-per-class rows and with the same prover protocol.
Result: no model's sycophancy readout moves beyond the random control, and
the significant effects are degradation rather than the probe's semantics —
Kimina's "proved" direction at +32 lowers provability of true steps by
raising type-check failures (the opposite sign of the residual-stream
effect), Herald breaks at both extremes. Only Goedel's "refute negatives more
often" trend recurs. The heads that read the verdict best are not, by
addition, the circuit that decides how faithfully to formalize.

### 7.4 Steering coverage — `leansteer/run/variants.py`

Five intervention families on the same rows and prover protocol: a layer
sweep (50 %, 75 %, 100 % depth and the selected layer), multi-layer steering
over the upper half, chain-of-thought-only steering (generated tokens only;
`positions="generated"` in the hooks), ITI σ-scaled head steering, and
projection ablation (`Intervention(kind="project")`, the dm component set to
the train mean or a class mean, random direction as control).  Scored by the
faithfulness index on the sycophancy rows, no family beats its random
control (thinking models ≤ +0.03; Herald's +0.05–0.08 is matched by random);
apparent "anti-sycophancy" numbers at single layers come with an equal drop
on true steps; multi-layer steering at α ±4 destroys the thinking models'
outputs where random directions do not; ablation leaves sycophancy and
Kimina's provability untouched; Kimina's provability effect survives
generation-only steering and lives at the last layer.  The sycophancy null
therefore holds across sites, layers, application windows, step-size
conventions and additive/subtractive interventions.

---

## 8. Robustness to the prover

**TL;DR.** Everything above uses DeepSeek-Prover-V2 as the arbiter. We
re-proved 1,500 held-out benchmark statements per model with
Goedel-Prover-V2-8B. Verdict agreement is 0.72–0.78 (0.88–0.93 on proved-vs-
not), the disagreements being mostly Proved/Refuted → Unknown. The probes read
the same thing under either prover: AUROC against the new labels is within
≈ 0.03 of AUROC against the benchmark labels on the same rows, for every model
and task.

### 8.1 Protocol — `leansteer/analysis/relabel.py`

- `relabel-select`: from each model's held-out problems (the `proved` split),
  sample 1,500 step variants uniformly over all verdicts except AF-Fail, and
  reconstruct the statement the benchmark proved: the raw response's theorem
  (true steps), the stored `sorry` statement (TC-Fail/Unknown), the proof's
  signature (Proved) or the un-negated negation proof (Refuted;
  `prove.py::unnegate_full`). 5,893 unique statements.
- `relabel-gen` / `relabel-verify`: the §7 machinery with
  `relabel.prover = goedel` (Qwen3-based; same prompt; no `<think>` tags in
  practice; 1 sample, 4,096-token cap because it runs to the cap on unprovable
  negations). Output under `runs/af_v1/relabel_goedel/`.
- `relabel-eval`: per model, the verdict confusion matrix; per task, AUROC of
  the *existing* direction (trained on benchmark labels) against benchmark
  labels and against the new labels on the same rows; 5-fold problem-grouped
  cross-validated probes trained on either label source; and the fixed
  direction on the rows where both provers agree.

### 8.2 Results

Fixed-direction AUROC, benchmark labels → Goedel-Prover labels: `proved`
0.72→0.71 (Goedel), 0.72→0.71 (Kimina), 0.67→0.65 (Herald), 0.71→0.72
(StepFun); `proved_strict` 0.83→0.81, 0.82→0.79, 0.87→0.84, 0.83→0.83;
`sycophancy` 0.85→0.87, 0.89→0.83, 0.90→0.89, 0.90→0.88. Cross-validated
probes trained on the new labels match those trained on the benchmark labels
(within 0.04). The one larger drop is Goedel `failing` (0.69→0.57), the task
already shown to be problem-level. (Fixed-direction numbers are higher than on
the full test set because this subset is uniformly sampled, not
class-balanced; the comparison of interest is bench-vs-new on identical rows.)

## 8b. Generalization beyond FaithformBench — `leansteer/data/external.py`, `run/external.py`, `analysis/external_eval.py`

**TL;DR (updated 2026-11-07).** Four external sets in the end. On the raw
sets the FaithformBench truth direction transferred weakly and retrained
probes barely beat text; two refinements changed that. Restricting PRM800K to
steps whose human rating agrees with an LLM judge's claim verdict (only
1,687 of 5,067 "−1" steps assert a false claim) lifts retrained truth probes
to 0.67–0.77 against text at 0.63 and provability probes to 0.76–0.85 against
0.58–0.64, with the benchmark directions transferring at 0.6–0.7 for
Goedel/StepFun. Enlarging miniF2F to 1,988 statements (1,970 corruptions)
makes the corruption readable at 0.67–0.71 against text at 0.64, still with
weak direction transfer. The representation generalizes; the direction must
be refitted; label quality and statement granularity set the gap over text.
Original summary: two external datasets were pushed through the same pipeline:
PRM800K's phase-2 test steps with their human correctness ratings (800 true,
800 false, 210 problems) and miniF2F's 488 informal statements corrupted with
the benchmark's own adversary prompt (Gemma-3-12B-it).  Activations, greedy
autoformalizations with the four models, DeepSeek-Prover verdicts and the
transfer evaluation (`ext-collect`, `ext-generate`, `ext-prove-*`,
`ext-eval`) all reuse the existing stages.  The FaithformBench truth
direction transfers weakly (PRM800K 0.51–0.66, miniF2F 0.50–0.57 without
retraining); probes retrained on PRM800K reach 0.70–0.74 but text reaches
0.69 there, and on miniF2F nothing reads the corruption above 0.62.
Provability-related verdict probes retrained on the new data do beat text
(`proved_strict` 0.72–0.77 vs 0.62–0.66 on PRM800K).  The representation is
real but benchmark-shaped: tuned to atomic steps and BrokenMath corruptions,
with PRM800K's labels meaning "rejected reasoning" rather than "false
proposition" and miniF2F's corruptions being single-number edits in long
problems that the models formalize faithfully (FPR 0.07–0.12).

---

## 9. Infrastructure and run layout

**TL;DR.** One YAML config, one CLI with resumable stages, one scratch tree per
run. GPU stages go through the host's `gpuq` queue and CPU stages through
`systemd-run --user`, because processes started from an interactive shell here
are killed after about two minutes.

### 9.1 Configuration and CLI

`leansteer/experiment.py::DEFAULTS` holds every knob; `configs/default.yaml`
overrides it for the reported run (`run_name: af_v1`) and `configs/smoke.yaml`
is a 96-row end-to-end test. Any entry can be overridden on the command line
with dotted keys (`--set steer.batch_size=8 prove.shard=1/2`). `Run` (same
file) owns the directory layout:

```
runs/<run>/<model>/acts/, acts_tf/          activations (§3.3)
runs/<run>/<model>/probes/, probes_tf/      per anchor/task: layers.csv, result.json, direction_L<k>.npz, transfer.json
runs/<run>/<model>/acts_heads/, probes_heads/ per-head store and head probes (heads.csv, heatmap.png) (§4.5)
runs/<run>/<model>/baselines/, modernbert/  §6
runs/<run>/<model>/steer/<task>/            generations.jsonl, typecheck.json, results.json (§5)
runs/<run>/prove/, prove_heads/             statements.json, candidates.jsonl, verified.json, results.json (§7, §7.3)
runs/<run>/<model>/steer_heads/<task>/      head-steered generations (§7.3)
runs/<run>/relabel_goedel/                  the same plus rows.json (§8)
runs/<run>/report/summary.md                everything, with plots
```

Stages (`leansteer/cli.py`): `data`, `download`, `collect`, `collect-tf`,
`probe`, `probe-tf`, `collect-heads`, `probe-heads`, `steer`, `steer-heads`, `prove-gen`, `prove-verify`, `prove-heads-gen`, `prove-heads-verify`, `baselines`,
`modernbert`, `relabel-select`, `relabel-gen`, `relabel-verify`,
`relabel-eval`, `report`.

### 9.2 Environments

- `leansteer` conda env (python 3.11, torch cu126, transformers 5) for
  everything except vLLM; the default pip torch wheel does not match the
  host's driver.
- `bmuq` env for `prove-gen` / `relabel-gen` (vLLM 0.13).
- Lean 4.33.1 via elan and the built Mathlib at
  `/scratch/datasets/lean-stockfish/mathlib`, used read-only through
  `LEAN_PATH`.
- Model weights come from the shared Hugging Face cache (`HF_HOME`).

### 9.3 Launchers

`scripts/gpuq_stage.sh <stage> <config> <card> <GB> [args]` submits a stage
beside one of your own cards; `scripts/cpu_stage.sh` runs a CPU stage as a
transient user unit; `scripts/run_all.sh` chains the basic pipeline. Jobs
attached to a card end when the card holder's 48-hour job ends; every stage is
resumable, so re-submitting is enough.

### 9.4 Tests

`tests/test_faithform.py` checks verdict normalisation, the four task
definitions, grouped splits and statement extraction; `tests/test_backend.py`
checks, on a tiny random Qwen2 model with a real tokenizer, that the step span
is found, that batched left-padded capture equals unbatched capture, that the
mean anchor equals the mean of per-token states, that a steering vector moves
the residual by exactly that vector downstream and not upstream, and that all
four tokenizers round-trip a prompt.

---

## 10. Conclusions and open ends

**TL;DR.** The autoformalizers carry a linearly readable "this step is true
and provable" signal before they write anything; it is not an artefact of
text, dataset or difficulty, it is shared across the proved/strict/sycophancy
distinctions, and it survives swapping the prover. It is causally usable for
one model (Kimina) on provability; it is not, at this layer and magnitude, a
lever on sycophancy for any model. Failure to formalize is a different, mostly
problem-level matter.

What this supports:

1. Verdict predictability before generation is real and model-internal
   (§4, §6), and what is read is mainly whether the step is *true* (§6.5),
   plus, among true steps, whether it is provable.
2. The pre-generation state is informative but far from decisive: the
   post-output state and the output text itself are much more predictive
   (§4.4), so the formalization process adds most of the information.
3. The sycophancy *readout* and the sycophancy *behaviour* come apart (§7,
   §7.3): knowing a step is corrupted does not, when amplified in the
   residual stream or in the heads that read it best, stop the model from
   repairing it.

What it does not settle:

- Steering has now been tried at several layers, multi-layer, during the
  chain of thought only, on heads with two step-size conventions, and as
  ablation (§7.3–7.4); what has not been tried is non-linear or learned
  interventions, or steering a different quantity than the probe directions.
- The Lean type-check proxy uses Mathlib v4.33.1, not the benchmark's version,
  and the prover runs use 2 samples (DeepSeek) or 1 (Goedel-Prover); verdicts
  are noisier than the benchmark's.
- The truth direction is benchmark-shaped (§8b): it does not transfer to
  PRM800K's human correctness labels or to miniF2F corruptions, and on those
  data probes add little beyond text for truth (they still do for
  provability verdicts).
- Teacher-forced probes exist for true steps only, by construction of the
  data; a regeneration of the corrupted steps' outputs with each model would
  remove that limit at the cost of a full generation pass.
- `failing` deserves a text-and-problem model rather than a probe; its
  ModernBERT baseline already beats the activation probe.
