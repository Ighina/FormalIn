# LeanSteer — background and verification guide

This document is the companion to the report: it states each result, gives the
numbers as they stand in the run artefacts, says which code produced them and
where the files are, and how to recompute them. It is **generated** by
`scripts/build_background.py` from the run `af_v1`; rerun it after
any stage and the tables update. Run root: `/scratch/users/iacopog/leansteer/runs/af_v1`. Human-written interpretation
is marked as such; everything in a table comes from a JSON file named in the
"verify" box below it.

Conventions: AUROC on held-out *problems* (30 % of problem ids, seed 0),
probe layer chosen on a validation split of the training problems, never on
test. "text" = TF-IDF (1–2-grams) logistic regression on the step text with the
same split. Model keys: goedel = Goedel-Formalizer-V2-8B, stepfun =
StepFun-Formalizer-7B, kimina = Kimina-Autoformalizer-7B, herald =
Herald_translator.

## 0. How to verify anything here

- Regenerate the full report with plots: `python -m leansteer.cli report -c configs/default.yaml`
  → `/scratch/users/iacopog/leansteer/runs/af_v1/report/summary.md`.
- Every stage is idempotent and resumable; rerunning a stage on existing
  artefacts recomputes its JSON without touching others. GPU stages on this
  host go through `scripts/gpuq_stage.sh`, CPU stages through `scripts/cpu_stage.sh`
  (see README §Running).
- Unit tests (`python -m pytest -q`, 13 tests) cover the task definitions,
  statement extraction, anchor capture, the steering hooks and tokenizer
  round-trips.
- Data file: `/home/iacopog/Downloads/45134_FaithformBench_Benchmark_Code And Data Supplement 3/Faithform-AF.csv`; loader
  `leansteer/data/faithform.py::load_af`; the task tables are printed by
  `python -m leansteer.cli data -c configs/default.yaml`.


## 1. Question and data

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


| task | positives | negatives | note |
|---|---|---|---|
| `proved` | true step, Proved | everything else | negatives subsampled to the positive count (`balance`) |
| `proved_strict` | true step, Proved | corrupted step, Refuted |  |
| `failing` | verdict ∉ {AF-Fail, TC-Fail} | AF-Fail or TC-Fail | Unknown counts as not failing |
| `sycophancy` | true step, Proved | corrupted step, Proved | the benchmark's sycophantic case |
| `perturbed_detect` | true step (any verdict) | corrupted step (any verdict) | control: a pure corruption detector; balanced |
| `perturbed_proved` | corrupted & Proved | corrupted & Refuted | control: sycophancy among corrupted steps only |
| `proved_unp`, `failing_unp` | as proved / failing | restricted to true steps | for teacher-forced anchors (§4.4) |

Task sizes (rows / positives / negatives / problems):

| model | task | n | pos | neg | problems |
|---|---|---|---|---|---|
| goedel | proved | 18684 | 9342 | 9342 | 1179 |
| goedel | proved_strict | 16109 | 9342 | 6767 | 1178 |
| goedel | failing | 25538 | 23947 | 1591 | 1179 |
| goedel | sycophancy | 12213 | 9342 | 2871 | 1178 |
| goedel | perturbed_proved | 9638 | 2871 | 6767 | 1170 |
| goedel | perturbed_detect | 25508 | 12754 | 12754 | 1179 |
| herald | proved | 10270 | 5135 | 5135 | 1178 |
| herald | proved_strict | 9488 | 5135 | 4353 | 1146 |
| herald | failing | 25538 | 17564 | 7974 | 1179 |
| herald | sycophancy | 7373 | 5135 | 2238 | 1122 |
| herald | perturbed_proved | 6591 | 2238 | 4353 | 1136 |
| herald | perturbed_detect | 25508 | 12754 | 12754 | 1179 |
| kimina | proved | 10542 | 5271 | 5271 | 1178 |
| kimina | proved_strict | 9343 | 5271 | 4072 | 1094 |
| kimina | failing | 25538 | 19084 | 6454 | 1179 |
| kimina | sycophancy | 7462 | 5271 | 2191 | 1081 |
| kimina | perturbed_proved | 6263 | 2191 | 4072 | 1071 |
| kimina | perturbed_detect | 25508 | 12754 | 12754 | 1179 |
| stepfun | proved | 13658 | 6829 | 6829 | 1179 |
| stepfun | proved_strict | 11913 | 6829 | 5084 | 1157 |
| stepfun | failing | 25538 | 20835 | 4703 | 1179 |
| stepfun | sycophancy | 9032 | 6829 | 2203 | 1145 |
| stepfun | perturbed_proved | 7287 | 2203 | 5084 | 1132 |
| stepfun | perturbed_detect | 25508 | 12754 | 12754 | 1179 |

- **code:** `leansteer/data/faithform.py` (`load_af`, `TASKS`, `build_task`, `problem_split`, `extract_statement`)
- **recompute:** `python -m leansteer.cli data -c configs/default.yaml`
- **tests:** `tests/test_faithform.py`

## 2. Pipeline map

**TL;DR.** One YAML config, one CLI, resumable stages, one scratch tree per run.


| stage (`python -m leansteer.cli …`) | code | writes (under run root) | section |
|---|---|---|---|
| `collect` | `run/collect.py::run_collect` | `<model>/acts/{prompt_last,step_last,step_mean}.npy`, `index.json`, `meta.jsonl` | §3 |
| `probe` | `analysis/probes.py::run_probes` | `<model>/probes/<anchor>/<task>/{result.json,layers.csv,direction_L*.npz}`, `transfer.json`, `summary.json` | §4 |
| `uncertainty` | `analysis/uncertainty.py` | `<model>/uncertainty/results.json` | §4.2 |
| `controls` | `analysis/controls.py` | `<model>/controls/results.json` | §5.2 |
| `baselines` | `analysis/baselines.py` | `<model>/baselines/{<task>.json,summary.json,features.parquet}` | §5.1 |
| `modernbert`, `modernbert-tf` | `analysis/modernbert.py` | `<model>/modernbert{,_tf}/summary.json` | §5.3, §4.4 |
| `collect-heads`, `probe-heads` | `run/collect.py::run_collect_heads`, `analysis/heads.py` | `<model>/acts_heads/`, `<model>/probes_heads/…/{result.json,heads.csv,heatmap.png}` | §6 |
| `collect-tf`, `probe-tf` | `run/collect.py::run_collect_tf`, `probes.py` (store=acts_tf) | `<model>/acts_tf/`, `<model>/probes_tf/` | §4.4 |
| `abstain` | `analysis/abstain.py` | `<model>/abstain/results.json` | §7 |
| `steer`, `steer-heads`, `steer-variant` | `run/steer.py`, `run/variants.py` | `<model>/steer{,_heads,_<variant>}/<task>/{generations.jsonl,results.json}` | §8 |
| `prove-gen`, `prove-verify` (+ `-heads-`, `-variant-`) | `run/prove.py`, `analysis/lean.py` | `prove{,_heads,_<variant>}/{statements.json,candidates.jsonl,verified.json,results.json}` | §8 |
| `relabel-select/gen/verify/eval` | `analysis/relabel.py`, `run/prove.py` | `relabel_goedel/{rows.json,statements.json,verified.json,results.json}` | §9 |
| `ext-collect/generate/prove-gen/prove-verify/eval` | `data/external.py`, `run/external.py`, `analysis/external_eval.py` | `ext_<name>/{steps.parquet,<model>/acts,<model>/generations.jsonl,prove/,verdicts.json,eval.json}` | §10 |
| `report` | `analysis/report.py` | `report/summary.md`, plots | all |

## 3. Models, prompts and what is read

**TL;DR.** The benchmark's verbatim prompts; a hooked HF backend capturing the
residual stream after every block at weighted token positions ("anchors") in
one prefill, plus additive / projection / per-head interventions for steering.

- Prompts: `leansteer/models/registry.py` (copied from the benchmark's
  `veriform/formalization/fine_tuned.py`; Goedel's statement abuts "Think
  before…" with no separator; StepFun gets a literal `<think>` appended).
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


## 4. Probes

**TL;DR.** Pre-generation probes read `proved`, `proved_strict` and
`sycophancy` at AUROC 0.66–0.80 (text 0.54–0.62), with ±0.02 intervals; one
shared direction across those three tasks; `failing` sits at the text
baseline. Post-output (teacher-forced) anchors read 0.82–0.96, but so does a
text model that sees the output.

### 4.1 Pre-generation probes (prompt_last and step_mean anchors)


| model | anchor | task | n | pos | layer | val | test AUROC | diff-means | TF-IDF | length |
|---|---|---|---|---|---|---|---|---|---|---|
| goedel | prompt_last | proved | 18684 | 9342 | 24 | 0.711 | **0.723** | 0.636 | 0.589 | 0.564 |
| goedel | step_mean | proved | 18684 | 9342 | 36 | 0.729 | **0.754** | 0.579 | 0.589 | 0.564 |
| goedel | prompt_last | proved_strict | 16109 | 9342 | 36 | 0.771 | **0.774** | 0.549 | 0.545 | 0.519 |
| goedel | step_mean | proved_strict | 16109 | 9342 | 36 | 0.796 | **0.803** | 0.579 | 0.545 | 0.519 |
| goedel | prompt_last | failing | 25538 | 23947 | 36 | 0.681 | **0.674** | 0.659 | 0.677 | 0.615 |
| goedel | step_mean | failing | 25538 | 23947 | 32 | 0.670 | **0.611** | 0.711 | 0.677 | 0.615 |
| goedel | prompt_last | sycophancy | 12213 | 9342 | 24 | 0.796 | **0.780** | 0.639 | 0.618 | 0.594 |
| goedel | step_mean | sycophancy | 12213 | 9342 | 22 | 0.771 | **0.771** | 0.643 | 0.618 | 0.594 |
| stepfun | prompt_last | proved | 13658 | 6829 | 28 | 0.715 | **0.709** | 0.607 | 0.587 | 0.563 |
| stepfun | step_mean | proved | 13658 | 6829 | 18 | 0.678 | **0.677** | 0.608 | 0.587 | 0.563 |
| stepfun | prompt_last | proved_strict | 11913 | 6829 | 28 | 0.741 | **0.773** | 0.571 | 0.548 | 0.510 |
| stepfun | step_mean | proved_strict | 11913 | 6829 | 28 | 0.727 | **0.775** | 0.551 | 0.548 | 0.510 |
| stepfun | prompt_last | failing | 25538 | 20835 | 24 | 0.777 | **0.738** | 0.636 | 0.702 | 0.568 |
| stepfun | step_mean | failing | 25538 | 20835 | 10 | 0.741 | **0.715** | 0.708 | 0.702 | 0.568 |
| stepfun | prompt_last | sycophancy | 9032 | 6829 | 28 | 0.777 | **0.772** | 0.629 | 0.604 | 0.585 |
| stepfun | step_mean | sycophancy | 9032 | 6829 | 28 | 0.766 | **0.767** | 0.604 | 0.604 | 0.585 |
| kimina | prompt_last | proved | 10542 | 5271 | 28 | 0.670 | **0.660** | 0.575 | 0.594 | 0.557 |
| kimina | step_mean | proved | 10542 | 5271 | 22 | 0.672 | **0.639** | 0.627 | 0.594 | 0.557 |
| kimina | prompt_last | proved_strict | 9343 | 5271 | 28 | 0.687 | **0.696** | 0.544 | 0.536 | 0.495 |
| kimina | step_mean | proved_strict | 9343 | 5271 | 28 | 0.739 | **0.731** | 0.584 | 0.536 | 0.495 |
| kimina | prompt_last | failing | 25538 | 19084 | 14 | 0.715 | **0.681** | 0.675 | 0.677 | 0.548 |
| kimina | step_mean | failing | 25538 | 19084 | 24 | 0.703 | **0.683** | 0.697 | 0.677 | 0.548 |
| kimina | prompt_last | sycophancy | 7462 | 5271 | 28 | 0.688 | **0.692** | 0.569 | 0.572 | 0.564 |
| kimina | step_mean | sycophancy | 7462 | 5271 | 28 | 0.715 | **0.749** | 0.568 | 0.572 | 0.564 |
| herald | prompt_last | proved | 10270 | 5135 | 26 | 0.650 | **0.635** | 0.655 | 0.599 | 0.591 |
| herald | step_mean | proved | 10270 | 5135 | 28 | 0.680 | **0.688** | 0.653 | 0.599 | 0.591 |
| herald | prompt_last | proved_strict | 9488 | 5135 | 16 | 0.680 | **0.706** | 0.617 | 0.563 | 0.506 |
| herald | step_mean | proved_strict | 9488 | 5135 | 28 | 0.771 | **0.785** | 0.632 | 0.563 | 0.506 |
| herald | prompt_last | failing | 25538 | 17564 | 2 | 0.693 | **0.666** | 0.608 | 0.639 | 0.575 |
| herald | step_mean | failing | 25538 | 17564 | 8 | 0.674 | **0.638** | 0.668 | 0.639 | 0.575 |
| herald | prompt_last | sycophancy | 7373 | 5135 | 16 | 0.702 | **0.696** | 0.640 | 0.584 | 0.557 |
| herald | step_mean | sycophancy | 7373 | 5135 | 28 | 0.746 | **0.766** | 0.605 | 0.584 | 0.557 |

- **files:** `<model>/probes/<anchor>/<task>/result.json` (`selected`, `baselines`), per-layer `layers.csv`, directions `direction_L<k>.npz`
- **code:** `leansteer/analysis/probes.py::run_probes`, `fit_layer`, `text_baselines`
- **recompute:** `python -m leansteer.cli probe -c configs/default.yaml --models <m>` (CPU, ~5 min per model)
- **plots:** `report/auroc_<model>_<anchor>.png`

### 4.2 Uncertainty

Problem-clustered bootstrap (2000 draws over held-out problems) of the seed-0 test AUROC and refits over 5 problem splits at the selected layer.


| model | task | layer | AUROC seed 0 | bootstrap 95% CI | 5 seeds mean ± sd | TF-IDF mean ± sd |
|---|---|---|---|---|---|---|
| goedel | proved | 24 | 0.723 | [0.707, 0.738] | 0.729 ± 0.004 | 0.576 ± 0.008 |
| goedel | proved_strict | 36 | 0.774 | [0.760, 0.789] | 0.777 ± 0.003 | 0.545 ± 0.003 |
| goedel | failing | 36 | 0.674 | [0.632, 0.710] | 0.667 ± 0.008 | 0.649 ± 0.021 |
| goedel | sycophancy | 24 | 0.780 | [0.760, 0.800] | 0.782 ± 0.005 | 0.617 ± 0.008 |
| stepfun | proved | 28 | 0.709 | [0.693, 0.726] | 0.711 ± 0.002 | 0.592 ± 0.011 |
| stepfun | proved_strict | 28 | 0.773 | [0.758, 0.787] | 0.769 ± 0.007 | 0.548 ± 0.005 |
| stepfun | failing | 24 | 0.737 | [0.706, 0.768] | 0.742 ± 0.007 | 0.696 ± 0.008 |
| stepfun | sycophancy | 28 | 0.772 | [0.752, 0.791] | 0.776 ± 0.011 | 0.610 ± 0.005 |
| kimina | proved | 28 | 0.660 | [0.639, 0.682] | 0.659 ± 0.015 | 0.597 ± 0.010 |
| kimina | proved_strict | 28 | 0.696 | [0.676, 0.717] | 0.700 ± 0.008 | 0.538 ± 0.010 |
| kimina | failing | 14 | 0.681 | [0.653, 0.709] | 0.680 ± 0.010 | 0.674 ± 0.012 |
| kimina | sycophancy | 28 | 0.692 | [0.669, 0.715] | 0.684 ± 0.005 | 0.569 ± 0.005 |
| herald | proved | 26 | 0.635 | [0.614, 0.655] | 0.624 ± 0.011 | 0.589 ± 0.011 |
| herald | proved_strict | 16 | 0.706 | [0.687, 0.726] | 0.690 ± 0.011 | 0.555 ± 0.010 |
| herald | failing | 2 | 0.666 | [0.643, 0.689] | 0.670 ± 0.005 | 0.653 ± 0.013 |
| herald | sycophancy | 16 | 0.696 | [0.672, 0.719] | 0.704 ± 0.007 | 0.583 ± 0.007 |

- **file:** `<model>/uncertainty/results.json`
- **code:** `leansteer/analysis/uncertainty.py::run_uncertainty`, `cluster_bootstrap_auc`
- **recompute:** `python -m leansteer.cli uncertainty -c configs/default.yaml --models <m>`

### 4.3 Cross-task transfer (prompt_last; direction of the row task scored on the column task's test rows)


**goedel**

| from \ on | proved | proved_strict | failing | sycophancy | perturbed_detect | perturbed_proved |
|---|---|---|---|---|---|---|
| proved (L24) | 0.72 | 0.76 | 0.61 | 0.77 | 0.70 | 0.47 |
| proved_strict (L36) | 0.74 | 0.77 | 0.47 | 0.74 | 0.77 | 0.61 |
| failing (L36) | 0.55 | 0.48 | 0.67 | 0.51 | 0.49 | 0.46 |
| sycophancy (L24) | 0.70 | 0.67 | 0.56 | 0.78 | 0.70 | 0.30 |
| perturbed_detect (L36) | 0.74 | 0.83 | 0.52 | 0.82 | 0.78 | 0.50 |
| perturbed_proved (L28) | 0.48 | 0.59 | 0.47 | 0.27 | 0.50 | 0.69 |

**stepfun**

| from \ on | proved | proved_strict | failing | sycophancy | perturbed_detect | perturbed_proved |
|---|---|---|---|---|---|---|
| proved (L28) | 0.71 | 0.76 | 0.63 | 0.78 | 0.67 | 0.50 |
| proved_strict (L28) | 0.71 | 0.77 | 0.52 | 0.73 | 0.74 | 0.61 |
| failing (L24) | 0.59 | 0.48 | 0.74 | 0.51 | 0.50 | 0.48 |
| sycophancy (L28) | 0.67 | 0.68 | 0.53 | 0.77 | 0.69 | 0.31 |
| perturbed_detect (L28) | 0.69 | 0.83 | 0.52 | 0.83 | 0.76 | 0.49 |
| perturbed_proved (L28) | 0.47 | 0.62 | 0.47 | 0.26 | 0.50 | 0.66 |

**kimina**

| from \ on | proved | proved_strict | failing | sycophancy | perturbed_detect | perturbed_proved |
|---|---|---|---|---|---|---|
| proved (L28) | 0.66 | 0.72 | 0.62 | 0.76 | 0.62 | 0.50 |
| proved_strict (L28) | 0.68 | 0.70 | 0.50 | 0.70 | 0.67 | 0.62 |
| failing (L14) | 0.65 | 0.49 | 0.68 | 0.51 | 0.50 | 0.49 |
| sycophancy (L28) | 0.68 | 0.68 | 0.51 | 0.69 | 0.65 | 0.33 |
| perturbed_detect (L28) | 0.67 | 0.78 | 0.49 | 0.81 | 0.70 | 0.49 |
| perturbed_proved (L12) | 0.49 | 0.60 | 0.49 | 0.30 | 0.50 | 0.56 |

**herald**

| from \ on | proved | proved_strict | failing | sycophancy | perturbed_detect | perturbed_proved |
|---|---|---|---|---|---|---|
| proved (L26) | 0.64 | 0.72 | 0.61 | 0.67 | 0.59 | 0.52 |
| proved_strict (L16) | 0.72 | 0.71 | 0.50 | 0.67 | 0.67 | 0.66 |
| failing (L2) | 0.61 | 0.47 | 0.67 | 0.54 | 0.50 | 0.44 |
| sycophancy (L16) | 0.66 | 0.65 | 0.51 | 0.70 | 0.65 | 0.32 |
| perturbed_detect (L16) | 0.64 | 0.77 | 0.51 | 0.78 | 0.69 | 0.49 |
| perturbed_proved (L16) | 0.53 | 0.63 | 0.51 | 0.29 | 0.50 | 0.59 |

- **file:** `<model>/probes/prompt_last/transfer.json`
- **code:** `probes.py::cross_task_transfer`
- **recompute:** `python -m leansteer.cli transfer -c configs/default.yaml --models <m>`

_Interpretation (human):_ `proved`, `proved_strict` and `sycophancy` transfer to each other within a few hundredths — one shared direction; `failing` neither transfers nor receives transfer.


### 4.4 Teacher-forced anchors (true steps; the model's recorded output appended to the prompt)


| model | anchor | task | n | layer | probe | TF-IDF step | TF-IDF step+output | ModernBERT step+statement (3 seeds) |
|---|---|---|---|---|---|---|---|---|
| goedel | output_last | proved_unp | 12757 | 8 | **0.814** | 0.692 | 0.818 | 0.829 ± 0.004 |
| goedel | statement_mean | proved_unp | 12757 | 20 | **0.860** | 0.692 | 0.818 | 0.829 ± 0.004 |
| goedel | output_last | failing_unp | 12757 | 30 | **0.889** | 0.647 | 0.886 | 0.921 ± 0.008 |
| goedel | statement_mean | failing_unp | 12757 | 24 | **0.920** | 0.647 | 0.886 | 0.921 ± 0.008 |
| stepfun | output_last | proved_unp | 12722 | 8 | **0.795** | 0.666 | 0.814 | 0.837 ± 0.002 |
| stepfun | statement_mean | proved_unp | 12722 | 16 | **0.839** | 0.666 | 0.814 | 0.837 ± 0.002 |
| stepfun | output_last | failing_unp | 12722 | 4 | **0.879** | 0.692 | 0.895 | 0.933 ± 0.004 |
| stepfun | statement_mean | failing_unp | 12722 | 22 | **0.917** | 0.692 | 0.895 | 0.933 ± 0.004 |
| kimina | output_last | proved_unp | 12695 | 6 | **0.819** | 0.627 | 0.761 | 0.828 ± 0.002 |
| kimina | statement_mean | proved_unp | 12695 | 8 | **0.820** | 0.627 | 0.761 | 0.828 ± 0.002 |
| kimina | output_last | failing_unp | 12695 | 26 | **0.944** | 0.675 | 0.909 | 0.970 ± 0.003 |
| kimina | statement_mean | failing_unp | 12695 | 6 | **0.956** | 0.675 | 0.909 | 0.970 ± 0.003 |
| herald | output_last | proved_unp | 12784 | 20 | **0.853** | 0.632 | 0.727 | 0.846 ± 0.004 |
| herald | statement_mean | proved_unp | 12784 | 16 | **0.863** | 0.632 | 0.727 | 0.846 ± 0.004 |
| herald | output_last | failing_unp | 12784 | 6 | **0.901** | 0.641 | 0.777 | 0.915 ± 0.004 |
| herald | statement_mean | failing_unp | 12784 | 10 | **0.910** | 0.641 | 0.777 | 0.915 ± 0.004 |

- **files:** `<model>/probes_tf/<anchor>/<task>/result.json`, `<model>/modernbert_tf/summary.json`
- **code:** `run/collect.py::run_collect_tf` (anchors `output_last`, `statement_last`, `statement_mean`; outputs > 3072 tokens skipped), `analysis/modernbert.py` with `inputs="step_statement"`
- **recompute:** `collect-tf` (GPU) → `probe-tf` → `modernbert-tf` (GPU)

_Interpretation (human):_ once the formalization is written the verdict is largely in the text — a classifier that sees the step and the Lean theorem matches the post-output probe for every model. The pre-generation signal is what the activations add.


## 5. Is the probe reading text, difficulty, or perturbation?

**TL;DR.** Not text or difficulty: nuisance and problem-level covariates reach
0.53–0.72 on the verdict tasks, explain ≤ 13 % of the probe score, and the
probe keeps 0.65–0.76 after they are regressed out; AUROC is flat across
subsets and step types. `failing` is a problem-level property. But the
`sycophancy`/`proved_strict` probes largely read *corruption/truth*: a pure
corruption detector reads the state at 0.69–0.80 (text 0.54), separates the
sycophancy rows as well as the sycophancy probe, and inside corrupted steps
nothing predicts which get sycophantically proved (0.56–0.69 ≈ text).

### 5.1 Covariate groups (`analysis/baselines.py`)


| model | task | probe | ModernBERT (text) | nuisance | difficulty | covariates | oracle (7 other formalizers) | gain beyond cov. | probe after regressing cov. out | R² score on cov. | within n_others_proved strata |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goedel | proved | **0.723** | 0.607 | 0.590 | 0.630 | 0.633 | 0.921 | +0.107 | 0.686 | 0.06 | 0.624 |
| goedel | proved_strict | **0.774** | 0.567 | 0.535 | 0.539 | 0.544 | 0.982 | +0.231 | 0.763 | 0.00 | 0.681 |
| goedel | failing | **0.674** | 0.723 | 0.674 | 0.806 | 0.806 | 0.746 | +0.004 | 0.556 | 0.15 | 0.664 |
| goedel | sycophancy | **0.780** | 0.652 | 0.626 | 0.633 | 0.642 | 0.920 | +0.138 | 0.740 | 0.10 | 0.702 |
| stepfun | proved | **0.709** | 0.628 | 0.614 | 0.669 | 0.676 | 0.899 | +0.066 | 0.651 | 0.13 | 0.619 |
| stepfun | proved_strict | **0.773** | 0.557 | 0.529 | 0.540 | 0.545 | 0.985 | +0.220 | 0.762 | 0.01 | 0.676 |
| stepfun | failing | **0.737** | 0.772 | 0.674 | 0.855 | 0.860 | 0.753 | +0.009 | 0.571 | 0.22 | 0.730 |
| stepfun | sycophancy | **0.772** | 0.623 | 0.608 | 0.621 | 0.623 | 0.937 | +0.147 | 0.733 | 0.09 | 0.749 |
| kimina | proved | **0.660** | 0.645 | 0.627 | 0.717 | 0.719 | 0.874 | +0.020 | 0.594 | 0.08 | 0.566 |
| kimina | proved_strict | **0.696** | 0.561 | 0.549 | 0.573 | 0.582 | 0.978 | +0.116 | 0.678 | 0.01 | 0.644 |
| kimina | failing | **0.681** | 0.735 | 0.660 | 0.846 | 0.850 | 0.703 | +0.004 | 0.521 | 0.18 | 0.677 |
| kimina | sycophancy | **0.692** | 0.626 | 0.578 | 0.594 | 0.590 | 0.945 | +0.102 | 0.666 | 0.00 | 0.674 |
| herald | proved | **0.635** | 0.653 | 0.620 | 0.671 | 0.677 | 0.876 | +0.022 | 0.585 | 0.05 | 0.533 |
| herald | proved_strict | **0.706** | 0.577 | 0.529 | 0.554 | 0.559 | 0.989 | +0.144 | 0.691 | -0.01 | 0.643 |
| herald | failing | **0.666** | 0.715 | 0.652 | 0.748 | 0.757 | 0.699 | +0.006 | 0.547 | 0.22 | 0.660 |
| herald | sycophancy | **0.696** | 0.617 | 0.583 | 0.612 | 0.612 | 0.951 | +0.096 | 0.660 | 0.01 | 0.653 |

Groups: *nuisance* = length, numbers, operators, position in trace, anchor
entropy/top-prob, TF-IDF-SVD(50) of the step; *difficulty* = subset, number of
steps, leave-one-out proved/failing rate of the problem's other steps (this
model and pooled over all 8 formalizers), statement_type, proof_type;
*oracle* = the other 3 dedicated + 4 LLM formalizers' verdicts on the same
step (encodes the step's truth; reported as a ceiling, not residualised). Full
per-task JSON also holds within-subset AUROC (and within-subset *training*),
within statement_type, and correlations with single covariates.


- **files:** `<model>/baselines/<task>.json`, `summary.json`; features in `features.parquet`
- **code:** `analysis/baselines.py::build_features` (groups), `run_baselines` (stacking, residualisation, strata)
- **metadata inputs:** `FaithformBench.csv` (statement_type, proof_type), `Faithform-LLM.csv` (LLM formalizer verdicts, joined on (problem_id, step_content))
- **recompute:** `python -m leansteer.cli baselines -c configs/default.yaml --models <m>`

### 5.2 Perturbation-confound controls (`analysis/controls.py`; prompt_last, sycophancy's selected layer)


| model | layer | within corrupted (Proved vs Refuted): proved_unp dir | … sycophancy dir | … perturbed dir | sycophancy rows: proved_unp dir | … perturbed dir | … own dir | cos(sycophancy, perturbed) | sycophancy AUROC w/o perturbed dir |
|---|---|---|---|---|---|---|---|---|---|
| goedel | 24 | 0.41 | 0.39 | 0.54 | 0.59 | 0.22 | 0.77 | -0.32 | 0.76 |
| stepfun | 28 | 0.45 | 0.38 | 0.53 | 0.55 | 0.22 | 0.76 | -0.38 | 0.75 |
| kimina | 28 | 0.45 | 0.44 | 0.51 | 0.57 | 0.28 | 0.69 | -0.35 | 0.66 |
| herald | 16 | 0.51 | 0.44 | 0.51 | 0.54 | 0.28 | 0.70 | -0.28 | 0.67 |

Per-layer probes trained *for* the control distinctions:

| model | corruption detector `perturbed_detect` | among corrupted: Proved vs Refuted `perturbed_proved` | `sycophancy` (reference) |
|---|---|---|---|
| goedel | 0.776 (text 0.535) | 0.686 (text 0.643) | 0.780 |
| stepfun | 0.761 (text 0.535) | 0.660 (text 0.620) | 0.772 |
| kimina | 0.699 (text 0.535) | 0.564 (text 0.600) | 0.692 |
| herald | 0.687 (text 0.535) | 0.591 (text 0.589) | 0.696 |

- **files:** `<model>/controls/results.json`; `<model>/probes/prompt_last/{perturbed_detect,perturbed_proved}/result.json`
- **code:** `analysis/controls.py::run_controls` (directions `proved_unp`, `perturbed`, `sycophancy`, `proved_strict`; eval sets incl. `within_perturbed`)
- **recompute:** `controls` and `probe --tasks perturbed_proved perturbed_detect`

_Interpretation (human):_ the corruption detector separates the sycophancy
rows as well as the sycophancy probe (sign flipped) and is anti-aligned with
it (cos −0.3 in 3.5–4k dimensions); inside corrupted steps no pre-trained
direction predicts which ones get sycophantically proved (0.39–0.54, chance or
inverted), and a probe trained for that question reaches 0.56–0.69 against
text at 0.59–0.64. The supportable claim is "the model knows the step is
false" (and, among true steps, whether it is provable), not "the model knows
it is about to repair the error".

### 5.3 ModernBERT on the step text (3 seeds)


| model | proved | proved_strict | failing | sycophancy |
|---|---|---|---|---|
| goedel | 0.607 ± 0.003 | 0.567 ± 0.006 | 0.723 ± 0.007 | 0.652 ± 0.004 |
| stepfun | 0.628 ± 0.001 | 0.557 ± 0.002 | 0.772 ± 0.006 | 0.623 ± 0.008 |
| kimina | 0.645 ± 0.005 | 0.561 ± 0.014 | 0.735 ± 0.006 | 0.626 ± 0.007 |
| herald | 0.653 ± 0.006 | 0.577 ± 0.015 | 0.715 ± 0.006 | 0.617 ± 0.001 |

- **file:** `<model>/modernbert/summary.json`
- **code:** `analysis/modernbert.py` (answerdotai/ModernBERT-base, 3 epochs, lr 3e-5, max 160 tokens, class-weighted BCE, checkpoint on validation AUROC)
- **recompute:** `modernbert` (GPU) with `--set modernbert.seeds=3`

## 6. Attention-head probes

**TL;DR.** One well-chosen late head matches the whole residual probe; the
signal is sparse (median head near chance); concatenating 4–64 heads beats the
residual probe by 0.05–0.11 on the verdict tasks. `failing` heads sit
mid-network and stay at the text/difficulty baselines.


| model | task | best single head | median / p90 head AUROC | heads > 0.6 | all heads of a layer | top-4 | top-16 | top-64 | residual probe |
|---|---|---|---|---|---|---|---|---|---|
| goedel | proved | L36H8 0.763 | 0.62 / 0.68 | 843/1152 | L36 0.782 | 0.802 | 0.805 | 0.786 | **L24 0.723** |
| goedel | proved_strict | L36H8 0.787 | 0.59 / 0.67 | 508/1152 | L36 0.833 | 0.843 | 0.860 | 0.854 | **L36 0.774** |
| goedel | failing | L28H28 0.733 | 0.71 / 0.73 | 1150/1152 | L32 0.644 | 0.702 | 0.650 | 0.672 | **L36 0.674** |
| goedel | sycophancy | L23H15 0.790 | 0.65 / 0.71 | 1125/1152 | L23 0.837 | 0.820 | 0.844 | 0.869 | **L24 0.780** |
| stepfun | proved | L24H26 0.672 | 0.63 / 0.67 | 649/784 | L21 0.717 | 0.722 | 0.726 | 0.731 | **L28 0.709** |
| stepfun | proved_strict | L28H8 0.727 | 0.59 / 0.67 | 350/784 | L21 0.778 | 0.779 | 0.812 | 0.824 | **L28 0.773** |
| stepfun | failing | L26H20 0.773 | 0.72 / 0.75 | 781/784 | L20 0.729 | 0.769 | 0.761 | 0.731 | **L24 0.738** |
| stepfun | sycophancy | L16H20 0.763 | 0.64 / 0.71 | 698/784 | L20 0.797 | 0.835 | 0.853 | 0.851 | **L28 0.772** |
| kimina | proved | L24H11 0.682 | 0.63 / 0.65 | 700/784 | L24 0.649 | 0.732 | 0.703 | 0.694 | **L28 0.660** |
| kimina | proved_strict | L28H22 0.702 | 0.56 / 0.60 | 60/784 | L28 0.713 | 0.746 | 0.771 | 0.771 | **L28 0.696** |
| kimina | failing | L8H26 0.710 | 0.68 / 0.70 | 781/784 | L14 0.676 | 0.709 | 0.703 | 0.691 | **L14 0.681** |
| kimina | sycophancy | L28H22 0.707 | 0.60 / 0.63 | 415/784 | L28 0.710 | 0.800 | 0.791 | 0.791 | **L28 0.692** |
| herald | proved | L15H16 0.671 | 0.63 / 0.66 | 883/960 | L29 0.658 | 0.711 | 0.706 | 0.698 | **L26 0.635** |
| herald | proved_strict | L16H9 0.682 | 0.59 / 0.64 | 371/960 | L29 0.739 | 0.807 | 0.811 | 0.814 | **L16 0.706** |
| herald | failing | L9H14 0.691 | 0.66 / 0.68 | 958/960 | L4 0.637 | 0.693 | 0.674 | 0.635 | **L2 0.666** |
| herald | sycophancy | L16H9 0.742 | 0.61 / 0.66 | 742/960 | L16 0.749 | 0.817 | 0.808 | 0.795 | **L16 0.696** |

- **files:** `<model>/probes_heads/prompt_last/<task>/{result.json,heads.csv,layers.csv,heatmap.png}`
- **code:** `run/collect.py::run_collect_heads` (pre-hook on `o_proj`; `backend.capture(component="heads")`), `analysis/heads.py::run_head_probes`
- **recompute:** `collect-heads` (GPU, ~2 min per model) → `probe-heads` (CPU)
- **plots:** `report/heads_<model>_prompt_last_<task>.png`

## 7. Abstention: the applied use of the probe

**TL;DR.** Refusing the 20 % of steps the sycophancy direction scores lowest
lowers the benchmark's UFLB (unfaithfulness lower bound, lower is better) for
every model and removes 40–54 % of sycophantic cases (random 21 %, a text
corruption detector 21–24 %).


| model | no abstention | sycophancy dir q=0.2 | sycophancy dir q=0.3 | proved dir q=0.2 | corruption detector q=0.2 | TF-IDF detector q=0.2 | random q=0.2 |
|---|---|---|---|---|---|---|---|
| goedel | 0.197 (FNR 0.04, FPR 0.23, AFFR 0.06) | 0.166 (46%) | 0.146 (64%) | 0.180 (35%) | 0.195 (36%) | 0.204 (21%) | 0.195 (22%) |
| stepfun | 0.316 (FNR 0.10, FPR 0.19, AFFR 0.17) | 0.276 (54%) | 0.252 (75%) | 0.290 (28%) | 0.304 (39%) | 0.323 (23%) | 0.315 (21%) |
| kimina | 0.408 (FNR 0.14, FPR 0.17, AFFR 0.25) | 0.382 (42%) | 0.354 (69%) | 0.385 (19%) | 0.406 (35%) | 0.416 (22%) | 0.410 (21%) |
| herald | 0.484 (FNR 0.15, FPR 0.17, AFFR 0.33) | 0.454 (40%) | 0.426 (71%) | 0.457 (19%) | 0.477 (32%) | 0.483 (24%) | 0.483 (21%) |

Cells: UFLB on the remaining held-out rows (share of sycophantic cases removed). UFLB = ½FNR + ½FPR + AFFR over all step variants of the held-out problems, using the benchmark's own verdicts (FaithformBench final metric, arXiv:2608.10916; lower is better).


- **file:** `<model>/abstain/results.json` (curves for q ∈ {0, .05, .1, .2, .3, .5})
- **code:** `analysis/abstain.py::run_abstain`, `metrics`
- **recompute:** `python -m leansteer.cli abstain -c configs/default.yaml --models <m>`

## 8. Steering, judged by the prover

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


28274 unique statements proved; verdict counts {'Proved': 12393, 'Refuted': 5108, 'Unknown': 6545, 'TC_Fail': 4228}.

| model | task | steps | direction | outcome | n | rate at min α → 0 → max α | effect over range (perm. p) | random (min α / max α) |
|---|---|---|---|---|---|---|---|---|
| goedel | proved | neg | dm | Proved | 300 | 0.31 (-8) → 0.32 → 0.27 (+8) | -0.043 (p=0.064) | 0.28 / 0.31 |
| goedel | proved | neg | dm | Refuted | 300 | 0.29 (-8) → 0.35 → 0.40 (+8) | **+0.117 (p=0.000)** | 0.33 / 0.35 |
| goedel | proved | neg | dm | TC_Fail | 300 | 0.12 (-8) → 0.09 → 0.08 (+8) | -0.041 (p=0.018) | 0.09 / 0.10 |
| goedel | proved | true | dm | Proved | 300 | 0.77 (-8) → 0.79 → 0.75 (+8) | -0.021 (p=0.355) | 0.81 / 0.80 |
| goedel | proved | true | dm | Refuted | 300 | 0.04 (-8) → 0.04 → 0.08 (+8) | +0.032 (p=0.020) | 0.04 / 0.04 |
| goedel | proved | true | dm | TC_Fail | 300 | 0.03 (-8) → 0.03 → 0.02 (+8) | -0.004 (p=0.727) | 0.05 / 0.03 |
| goedel | sycophancy | neg | dm | Proved | 300 | 0.41 (-8) → 0.37 → 0.38 (+8) | -0.047 (p=0.099) | 0.35 / 0.37 |
| goedel | sycophancy | neg | dm | Refuted | 300 | 0.26 (-8) → 0.31 → 0.32 (+8) | **+0.079 (p=0.001)** | 0.30 / 0.31 |
| goedel | sycophancy | neg | dm | TC_Fail | 300 | 0.07 (-8) → 0.06 → 0.06 (+8) | -0.012 (p=0.456) | 0.06 / 0.05 |
| goedel | sycophancy | true | dm | Proved | 294 | 0.74 (-8) → 0.79 → 0.74 (+8) | +0.001 (p=0.978) | 0.76 / 0.77 |
| goedel | sycophancy | true | dm | Refuted | 294 | 0.08 (-8) → 0.07 → 0.09 (+8) | +0.016 (p=0.332) | 0.06 / 0.08 |
| goedel | sycophancy | true | dm | TC_Fail | 294 | 0.05 (-8) → 0.02 → 0.05 (+8) | +0.004 (p=0.767) | 0.04 / 0.06 |
| herald | proved | neg | dm | Proved | 300 | 0.16 (-8) → 0.16 → 0.15 (+8) | +0.001 (p=0.962) | 0.14 / 0.18 |
| herald | proved | neg | dm | Refuted | 300 | 0.16 (-8) → 0.24 → 0.18 (+8) | +0.022 (p=0.340) | 0.19 / 0.20 |
| herald | proved | neg | dm | TC_Fail | 300 | 0.32 (-8) → 0.27 → 0.23 (+8) | **-0.067 (p=0.005)** | 0.34 / 0.26 |
| herald | proved | true | dm | Proved | 300 | 0.44 (-8) → 0.53 → 0.45 (+8) | +0.023 (p=0.378) | 0.42 / 0.50 |
| herald | proved | true | dm | Refuted | 300 | 0.07 (-8) → 0.08 → 0.07 (+8) | -0.002 (p=0.917) | 0.08 / 0.06 |
| herald | proved | true | dm | TC_Fail | 300 | 0.23 (-8) → 0.19 → 0.17 (+8) | **-0.058 (p=0.004)** | 0.28 / 0.18 |
| herald | sycophancy | neg | dm | Proved | 300 | 0.20 (-8) → 0.29 → 0.27 (+8) | +0.039 (p=0.139) | 0.24 / 0.26 |
| herald | sycophancy | neg | dm | Refuted | 300 | 0.21 (-8) → 0.23 → 0.22 (+8) | +0.028 (p=0.274) | 0.23 / 0.22 |
| herald | sycophancy | neg | dm | TC_Fail | 300 | 0.14 (-8) → 0.22 → 0.24 (+8) | **+0.107 (p=0.000)** | 0.25 / 0.19 |
| herald | sycophancy | true | dm | Proved | 293 | 0.39 (-8) → 0.47 → 0.53 (+8) | **+0.124 (p=0.000)** | 0.51 / 0.51 |
| herald | sycophancy | true | dm | Refuted | 293 | 0.09 (-8) → 0.10 → 0.07 (+8) | -0.012 (p=0.491) | 0.06 / 0.08 |
| herald | sycophancy | true | dm | TC_Fail | 293 | 0.18 (-8) → 0.17 → 0.16 (+8) | -0.007 (p=0.827) | 0.16 / 0.16 |
| kimina | proved | neg | dm | Proved | 157 | 0.15 (-8) → 0.22 → 0.28 (+8) | **+0.138 (p=0.000)** | 0.22 / 0.20 |
| kimina | proved | neg | dm | Refuted | 157 | 0.18 (-8) → 0.23 → 0.28 (+8) | **+0.097 (p=0.000)** | 0.28 / 0.24 |
| kimina | proved | neg | dm | TC_Fail | 157 | 0.36 (-8) → 0.16 → 0.13 (+8) | **-0.223 (p=0.000)** | 0.17 / 0.24 |
| kimina | proved | true | dm | Proved | 155 | 0.38 (-8) → 0.61 → 0.63 (+8) | **+0.233 (p=0.000)** | 0.60 / 0.57 |
| kimina | proved | true | dm | Refuted | 155 | 0.15 (-8) → 0.14 → 0.10 (+8) | **-0.085 (p=0.000)** | 0.11 / 0.15 |
| kimina | proved | true | dm | TC_Fail | 155 | 0.32 (-8) → 0.05 → 0.04 (+8) | **-0.241 (p=0.000)** | 0.05 / 0.06 |
| kimina | sycophancy | neg | dm | Proved | 300 | 0.32 (-8) → 0.32 → 0.33 (+8) | +0.016 (p=0.355) | 0.35 / 0.32 |
| kimina | sycophancy | neg | dm | Refuted | 300 | 0.36 (-8) → 0.39 → 0.39 (+8) | +0.040 (p=0.026) | 0.37 / 0.35 |
| kimina | sycophancy | neg | dm | TC_Fail | 300 | 0.06 (-8) → 0.05 → 0.05 (+8) | -0.016 (p=0.072) | 0.04 / 0.09 |
| kimina | sycophancy | true | dm | Proved | 297 | 0.64 (-8) → 0.64 → 0.61 (+8) | -0.031 (p=0.049) | 0.65 / 0.62 |
| kimina | sycophancy | true | dm | Refuted | 297 | 0.10 (-8) → 0.12 → 0.13 (+8) | +0.023 (p=0.076) | 0.13 / 0.12 |
| kimina | sycophancy | true | dm | TC_Fail | 297 | 0.05 (-8) → 0.04 → 0.05 (+8) | +0.004 (p=0.701) | 0.03 / 0.07 |
| stepfun | proved | neg | dm | Proved | 300 | 0.28 (-8) → 0.23 → 0.25 (+8) | -0.033 (p=0.108) | 0.25 / 0.25 |
| stepfun | proved | neg | dm | Refuted | 300 | 0.29 (-8) → 0.27 → 0.26 (+8) | -0.027 (p=0.225) | 0.30 / 0.28 |
| stepfun | proved | neg | dm | TC_Fail | 300 | 0.15 (-8) → 0.14 → 0.16 (+8) | +0.017 (p=0.361) | 0.17 / 0.15 |
| stepfun | proved | true | dm | Proved | 300 | 0.69 (-8) → 0.68 → 0.69 (+8) | +0.017 (p=0.459) | 0.72 / 0.68 |
| stepfun | proved | true | dm | Refuted | 300 | 0.08 (-8) → 0.08 → 0.07 (+8) | +0.000 (p=1.000) | 0.08 / 0.09 |
| stepfun | proved | true | dm | TC_Fail | 300 | 0.07 (-8) → 0.06 → 0.06 (+8) | -0.004 (p=0.790) | 0.06 / 0.06 |
| stepfun | sycophancy | neg | dm | Proved | 300 | 0.36 (-8) → 0.36 → 0.38 (+8) | +0.015 (p=0.586) | 0.37 / 0.37 |
| stepfun | sycophancy | neg | dm | Refuted | 300 | 0.35 (-8) → 0.30 → 0.32 (+8) | -0.021 (p=0.405) | 0.33 / 0.33 |
| stepfun | sycophancy | neg | dm | TC_Fail | 300 | 0.09 (-8) → 0.12 → 0.06 (+8) | -0.011 (p=0.528) | 0.11 / 0.10 |
| stepfun | sycophancy | true | dm | Proved | 292 | 0.68 (-8) → 0.69 → 0.70 (+8) | -0.007 (p=0.779) | 0.68 / 0.68 |
| stepfun | sycophancy | true | dm | Refuted | 292 | 0.07 (-8) → 0.09 → 0.08 (+8) | +0.022 (p=0.229) | 0.10 / 0.07 |
| stepfun | sycophancy | true | dm | TC_Fail | 292 | 0.08 (-8) → 0.08 → 0.09 (+8) | +0.005 (p=0.744) | 0.09 / 0.08 |

- **files:** `prove/results.json` (`table` per cell, `trends`), `prove/verified.json`, `<model>/steer/<task>/generations.jsonl`
- **code:** `run/steer.py`, `run/prove.py` (`run_prove_gen`, `run_prove_verify`, `dose_response`), `analysis/lean.py`
- **recompute:** `steer` (GPU) → `prove-gen` (GPU, bmuq env, `prove.shard=k/n`) → `prove-verify` (CPU)
- **note:** n ≈ 300 cells are the 300-per-class rerun (alphas ±4, ±8); n ≈ 150 cells keep the ±2 grid

### 8.3 Attention-head steering (top-16 heads, α in units of each head's class-mean distance)


14057 unique statements.

| model | task | steps | direction | outcome | n | rate at min α → 0 → max α | effect over range (perm. p) | random (min α / max α) |
|---|---|---|---|---|---|---|---|---|
| goedel | proved | neg | dm | Proved | 150 | 0.27 (-32) → 0.33 → 0.29 (+32) | +0.024 (p=0.453) | 0.29 / 0.33 |
| goedel | proved | neg | dm | Refuted | 150 | 0.25 (-32) → 0.33 → 0.36 (+32) | **+0.108 (p=0.002)** | 0.31 / 0.26 |
| goedel | proved | neg | dm | TC_Fail | 150 | 0.15 (-32) → 0.09 → 0.11 (+32) | -0.023 (p=0.209) | 0.09 / 0.14 |
| goedel | proved | true | dm | Proved | 150 | 0.61 (-32) → 0.77 → 0.67 (+32) | +0.011 (p=0.728) | 0.77 / 0.79 |
| goedel | proved | true | dm | Refuted | 150 | 0.12 (-32) → 0.03 → 0.10 (+32) | +0.015 (p=0.495) | 0.04 / 0.02 |
| goedel | proved | true | dm | TC_Fail | 150 | 0.08 (-32) → 0.03 → 0.03 (+32) | **-0.038 (p=0.004)** | 0.03 / 0.02 |
| goedel | sycophancy | neg | dm | Proved | 150 | 0.29 (-32) → 0.38 → 0.31 (+32) | -0.046 (p=0.265) | 0.35 / 0.37 |
| goedel | sycophancy | neg | dm | Refuted | 150 | 0.11 (-32) → 0.32 → 0.27 (+32) | **+0.168 (p=0.000)** | 0.24 / 0.28 |
| goedel | sycophancy | neg | dm | TC_Fail | 150 | 0.17 (-32) → 0.07 → 0.09 (+32) | **-0.065 (p=0.004)** | 0.11 / 0.05 |
| goedel | sycophancy | true | dm | Proved | 147 | 0.53 (-32) → 0.79 → 0.55 (+32) | -0.005 (p=0.880) | 0.67 / 0.75 |
| goedel | sycophancy | true | dm | Refuted | 147 | 0.10 (-32) → 0.07 → 0.11 (+32) | +0.029 (p=0.274) | 0.07 / 0.08 |
| goedel | sycophancy | true | dm | TC_Fail | 147 | 0.15 (-32) → 0.03 → 0.09 (+32) | -0.052 (p=0.011) | 0.07 / 0.02 |
| herald | proved | neg | dm | Proved | 150 | 0.17 (-32) → 0.10 → 0.07 (+32) | **-0.113 (p=0.001)** | 0.15 / 0.13 |
| herald | proved | neg | dm | Refuted | 150 | 0.19 (-32) → 0.20 → 0.19 (+32) | +0.027 (p=0.427) | 0.18 / 0.15 |
| herald | proved | neg | dm | TC_Fail | 150 | 0.14 (-32) → 0.29 → 0.37 (+32) | **+0.194 (p=0.000)** | 0.33 / 0.31 |
| herald | proved | true | dm | Proved | 150 | 0.45 (-32) → 0.50 → 0.32 (+32) | **-0.137 (p=0.001)** | 0.43 / 0.49 |
| herald | proved | true | dm | Refuted | 150 | 0.06 (-32) → 0.07 → 0.08 (+32) | +0.037 (p=0.106) | 0.10 / 0.07 |
| herald | proved | true | dm | TC_Fail | 150 | 0.23 (-32) → 0.21 → 0.21 (+32) | +0.015 (p=0.625) | 0.24 / 0.24 |
| herald | sycophancy | neg | dm | Proved | 150 | 0.03 (-32) → 0.29 → 0.23 (+32) | **+0.175 (p=0.000)** | 0.16 / 0.27 |
| herald | sycophancy | neg | dm | Refuted | 150 | 0.03 (-32) → 0.21 → 0.26 (+32) | **+0.198 (p=0.000)** | 0.15 / 0.14 |
| herald | sycophancy | neg | dm | TC_Fail | 150 | 0.18 (-32) → 0.25 → 0.26 (+32) | +0.052 (p=0.167) | 0.41 / 0.24 |
| herald | sycophancy | true | dm | Proved | 148 | 0.13 (-32) → 0.48 → 0.30 (+32) | **+0.152 (p=0.001)** | 0.34 / 0.41 |
| herald | sycophancy | true | dm | Refuted | 148 | 0.03 (-32) → 0.09 → 0.24 (+32) | **+0.188 (p=0.000)** | 0.10 / 0.09 |
| herald | sycophancy | true | dm | TC_Fail | 148 | 0.17 (-32) → 0.16 → 0.16 (+32) | -0.042 (p=0.208) | 0.27 / 0.18 |
| kimina | proved | neg | dm | Proved | 150 | 0.25 (-32) → 0.23 → 0.19 (+32) | -0.058 (p=0.021) | 0.22 / 0.23 |
| kimina | proved | neg | dm | Refuted | 150 | 0.28 (-32) → 0.23 → 0.21 (+32) | **-0.076 (p=0.002)** | 0.27 / 0.28 |
| kimina | proved | neg | dm | TC_Fail | 150 | 0.15 (-32) → 0.15 → 0.29 (+32) | **+0.131 (p=0.000)** | 0.22 / 0.18 |
| kimina | proved | true | dm | Proved | 150 | 0.57 (-32) → 0.60 → 0.47 (+32) | **-0.105 (p=0.000)** | 0.59 / 0.55 |
| kimina | proved | true | dm | Refuted | 150 | 0.15 (-32) → 0.15 → 0.12 (+32) | -0.018 (p=0.454) | 0.11 / 0.19 |
| kimina | proved | true | dm | TC_Fail | 150 | 0.06 (-32) → 0.05 → 0.19 (+32) | **+0.118 (p=0.000)** | 0.09 / 0.04 |
| kimina | sycophancy | neg | dm | Proved | 150 | 0.35 (-32) → 0.33 → 0.33 (+32) | -0.025 (p=0.320) | 0.32 / 0.35 |
| kimina | sycophancy | neg | dm | Refuted | 150 | 0.38 (-32) → 0.37 → 0.39 (+32) | +0.022 (p=0.396) | 0.37 / 0.35 |
| kimina | sycophancy | neg | dm | TC_Fail | 150 | 0.05 (-32) → 0.03 → 0.03 (+32) | -0.011 (p=0.231) | 0.04 / 0.04 |
| kimina | sycophancy | true | dm | Proved | 149 | 0.60 (-32) → 0.62 → 0.60 (+32) | -0.005 (p=0.794) | 0.63 / 0.63 |
| kimina | sycophancy | true | dm | Refuted | 149 | 0.13 (-32) → 0.15 → 0.14 (+32) | +0.010 (p=0.626) | 0.17 / 0.11 |
| kimina | sycophancy | true | dm | TC_Fail | 149 | 0.04 (-32) → 0.03 → 0.03 (+32) | -0.014 (p=0.231) | 0.02 / 0.03 |
| stepfun | proved | neg | dm | Proved | 150 | 0.24 (-32) → 0.21 → 0.21 (+32) | -0.025 (p=0.338) | 0.22 / 0.25 |
| stepfun | proved | neg | dm | Refuted | 150 | 0.29 (-32) → 0.25 → 0.28 (+32) | -0.023 (p=0.456) | 0.27 / 0.26 |
| stepfun | proved | neg | dm | TC_Fail | 150 | 0.13 (-32) → 0.15 → 0.18 (+32) | +0.061 (p=0.010) | 0.17 / 0.16 |
| stepfun | proved | true | dm | Proved | 150 | 0.65 (-32) → 0.69 → 0.71 (+32) | +0.015 (p=0.640) | 0.68 / 0.72 |
| stepfun | proved | true | dm | Refuted | 150 | 0.06 (-32) → 0.07 → 0.09 (+32) | +0.028 (p=0.203) | 0.07 / 0.05 |
| stepfun | proved | true | dm | TC_Fail | 150 | 0.07 (-32) → 0.06 → 0.05 (+32) | -0.011 (p=0.513) | 0.08 / 0.05 |
| stepfun | sycophancy | neg | dm | Proved | 150 | 0.29 (-32) → 0.37 → 0.35 (+32) | +0.067 (p=0.045) | 0.32 / 0.42 |
| stepfun | sycophancy | neg | dm | Refuted | 150 | 0.35 (-32) → 0.32 → 0.35 (+32) | +0.005 (p=0.890) | 0.30 / 0.31 |
| stepfun | sycophancy | neg | dm | TC_Fail | 150 | 0.12 (-32) → 0.13 → 0.11 (+32) | -0.018 (p=0.408) | 0.15 / 0.07 |
| stepfun | sycophancy | true | dm | Proved | 146 | 0.60 (-32) → 0.68 → 0.64 (+32) | +0.029 (p=0.376) | 0.65 / 0.68 |
| stepfun | sycophancy | true | dm | Refuted | 146 | 0.09 (-32) → 0.12 → 0.10 (+32) | +0.017 (p=0.540) | 0.10 / 0.10 |
| stepfun | sycophancy | true | dm | TC_Fail | 146 | 0.13 (-32) → 0.08 → 0.06 (+32) | **-0.067 (p=0.002)** | 0.11 / 0.05 |

- **files:** `prove_heads/results.json`, `<model>/steer_heads/<task>/{generations.jsonl,steer_heads_k16.npz}`
- **code:** `run/steer.py::run_steer_heads`, `head_class_stats`; `backend.py` `head_steer` hook
- **recompute:** `steer-heads` → `prove-heads-gen` → `prove-heads-verify`

### 8.4 Steering variants (`run/variants.py`; 150 steps per class)


**`layer_sweep`** (12297 statements; Proved outcome only, all directions):

| model | task | steps | direction | outcome | n | rate at min α → 0 → max α | effect over range (perm. p) | random (min α / max α) |
|---|---|---|---|---|---|---|---|---|
| goedel | proved | neg | dm_L18 | Proved | 150 | 0.21 (-8) → 0.33 → 0.21 (+8) | -0.007 (p=0.804) | 0.18 / 0.18 |
| goedel | proved | neg | dm_L24 | Proved | 150 | 0.31 (-8) → 0.33 → 0.27 (+8) | -0.040 (p=0.314) | 0.21 / 0.21 |
| goedel | proved | neg | dm_L27 | Proved | 150 | 0.18 (-8) → 0.33 → 0.18 (+8) | +0.000 (p=1.000) | 0.23 / 0.23 |
| goedel | proved | neg | dm_L36 | Proved | 150 | 0.21 (-8) → 0.33 → 0.18 (+8) | -0.033 (p=0.376) | 0.19 / 0.19 |
| goedel | proved | true | dm_L18 | Proved | 150 | 0.59 (-8) → 0.77 → 0.65 (+8) | +0.060 (p=0.127) | 0.65 / 0.65 |
| goedel | proved | true | dm_L24 | Proved | 150 | 0.69 (-8) → 0.77 → 0.67 (+8) | -0.020 (p=0.685) | 0.59 / 0.59 |
| goedel | proved | true | dm_L27 | Proved | 150 | 0.53 (-8) → 0.77 → 0.59 (+8) | +0.060 (p=0.227) | 0.52 / 0.52 |
| goedel | proved | true | dm_L36 | Proved | 150 | 0.53 (-8) → 0.77 → 0.59 (+8) | +0.060 (p=0.211) | 0.65 / 0.65 |
| goedel | sycophancy | neg | dm_L18 | Proved | 150 | 0.23 (-8) → 0.38 → 0.37 (+8) | **+0.140 (p=0.003)** | 0.29 / 0.29 |
| goedel | sycophancy | neg | dm_L24 | Proved | 150 | 0.37 (-8) → 0.38 → 0.41 (+8) | +0.033 (p=0.494) | 0.40 / 0.40 |
| goedel | sycophancy | neg | dm_L27 | Proved | 150 | 0.40 (-8) → 0.38 → 0.35 (+8) | -0.047 (p=0.325) | 0.37 / 0.37 |
| goedel | sycophancy | neg | dm_L36 | Proved | 150 | 0.38 (-8) → 0.38 → 0.39 (+8) | +0.007 (p=0.933) | 0.34 / 0.34 |
| goedel | sycophancy | true | dm_L18 | Proved | 147 | 0.59 (-8) → 0.79 → 0.65 (+8) | +0.061 (p=0.182) | 0.78 / 0.78 |
| goedel | sycophancy | true | dm_L24 | Proved | 147 | 0.76 (-8) → 0.79 → 0.75 (+8) | -0.007 (p=0.926) | 0.81 / 0.81 |
| goedel | sycophancy | true | dm_L27 | Proved | 147 | 0.76 (-8) → 0.79 → 0.76 (+8) | +0.007 (p=0.928) | 0.77 / 0.77 |
| goedel | sycophancy | true | dm_L36 | Proved | 147 | 0.80 (-8) → 0.79 → 0.79 (+8) | -0.007 (p=0.871) | 0.82 / 0.82 |
| herald | sycophancy | neg | dm_L15 | Proved | 150 | 0.29 (-8) → 0.29 → 0.25 (+8) | -0.033 (p=0.437) | 0.23 / 0.23 |
| herald | sycophancy | neg | dm_L16 | Proved | 150 | 0.23 (-8) → 0.29 → 0.25 (+8) | +0.020 (p=0.685) | 0.26 / 0.26 |
| herald | sycophancy | neg | dm_L22 | Proved | 150 | 0.11 (-8) → 0.29 → 0.27 (+8) | **+0.160 (p=0.000)** | 0.25 / 0.25 |
| herald | sycophancy | neg | dm_L30 | Proved | 150 | 0.09 (-8) → 0.29 → 0.15 (+8) | +0.067 (p=0.106) | 0.19 / 0.19 |
| herald | sycophancy | true | dm_L15 | Proved | 148 | 0.37 (-8) → 0.48 → 0.47 (+8) | +0.101 (p=0.022) | 0.46 / 0.46 |
| herald | sycophancy | true | dm_L16 | Proved | 148 | 0.41 (-8) → 0.48 → 0.55 (+8) | **+0.149 (p=0.002)** | 0.53 / 0.53 |
| herald | sycophancy | true | dm_L22 | Proved | 148 | 0.33 (-8) → 0.48 → 0.43 (+8) | +0.101 (p=0.034) | 0.49 / 0.49 |
| herald | sycophancy | true | dm_L30 | Proved | 148 | 0.25 (-8) → 0.48 → 0.36 (+8) | +0.115 (p=0.013) | 0.41 / 0.41 |
| kimina | proved | neg | dm_L14 | Proved | 150 | 0.25 (-8) → 0.23 → 0.22 (+8) | -0.033 (p=0.326) | 0.25 / 0.25 |
| kimina | proved | neg | dm_L21 | Proved | 150 | 0.24 (-8) → 0.23 → 0.23 (+8) | -0.013 (p=0.745) | 0.19 / 0.19 |
| kimina | proved | neg | dm_L28 | Proved | 150 | 0.15 (-8) → 0.23 → 0.27 (+8) | **+0.120 (p=0.000)** | 0.24 / 0.24 |
| kimina | proved | true | dm_L14 | Proved | 150 | 0.59 (-8) → 0.60 → 0.55 (+8) | -0.033 (p=0.396) | 0.59 / 0.59 |
| kimina | proved | true | dm_L21 | Proved | 150 | 0.57 (-8) → 0.60 → 0.52 (+8) | -0.047 (p=0.180) | 0.60 / 0.60 |
| kimina | proved | true | dm_L28 | Proved | 150 | 0.37 (-8) → 0.60 → 0.61 (+8) | **+0.240 (p=0.000)** | 0.57 / 0.57 |
| kimina | sycophancy | neg | dm_L14 | Proved | 150 | 0.31 (-8) → 0.33 → 0.35 (+8) | +0.047 (p=0.159) | 0.39 / 0.39 |
| kimina | sycophancy | neg | dm_L21 | Proved | 150 | 0.36 (-8) → 0.33 → 0.31 (+8) | -0.047 (p=0.102) | 0.37 / 0.37 |
| kimina | sycophancy | neg | dm_L28 | Proved | 150 | 0.34 (-8) → 0.33 → 0.31 (+8) | -0.027 (p=0.447) | 0.37 / 0.37 |
| kimina | sycophancy | true | dm_L14 | Proved | 149 | 0.60 (-8) → 0.62 → 0.62 (+8) | +0.020 (p=0.502) | 0.62 / 0.62 |
| kimina | sycophancy | true | dm_L21 | Proved | 149 | 0.61 (-8) → 0.62 → 0.60 (+8) | -0.013 (p=0.696) | 0.63 / 0.63 |
| kimina | sycophancy | true | dm_L28 | Proved | 149 | 0.60 (-8) → 0.62 → 0.58 (+8) | -0.020 (p=0.501) | 0.62 / 0.62 |
| stepfun | sycophancy | neg | dm_L14 | Proved | 150 | 0.35 (-8) → 0.37 → 0.35 (+8) | +0.007 (p=0.932) | 0.33 / 0.33 |
| stepfun | sycophancy | neg | dm_L21 | Proved | 150 | 0.41 (-8) → 0.37 → 0.38 (+8) | -0.027 (p=0.567) | 0.38 / 0.38 |
| stepfun | sycophancy | neg | dm_L28 | Proved | 150 | 0.36 (-8) → 0.37 → 0.37 (+8) | +0.013 (p=0.783) | 0.38 / 0.38 |
| stepfun | sycophancy | true | dm_L14 | Proved | 146 | 0.65 (-8) → 0.68 → 0.69 (+8) | +0.041 (p=0.305) | 0.73 / 0.73 |
| stepfun | sycophancy | true | dm_L21 | Proved | 146 | 0.68 (-8) → 0.68 → 0.68 (+8) | -0.007 (p=0.924) | 0.71 / 0.71 |
| stepfun | sycophancy | true | dm_L28 | Proved | 146 | 0.72 (-8) → 0.68 → 0.68 (+8) | -0.041 (p=0.310) | 0.73 / 0.73 |

**`multi_layer`** (4237 statements; Proved outcome only, all directions):

| model | task | steps | direction | outcome | n | rate at min α → 0 → max α | effect over range (perm. p) | random (min α / max α) |
|---|---|---|---|---|---|---|---|---|
| goedel | proved | neg | dm_multi | Proved | 150 | 0.00 (-4) → 0.33 → 0.00 (+4) | +0.000 (p=1.000) | 0.17 / 0.17 |
| goedel | proved | true | dm_multi | Proved | 150 | 0.00 (-4) → 0.77 → 0.00 (+4) | +0.000 (p=1.000) | 0.57 / 0.57 |
| goedel | sycophancy | neg | dm_multi | Proved | 150 | 0.00 (-4) → 0.38 → 0.00 (+4) | +0.000 (p=1.000) | 0.19 / 0.19 |
| goedel | sycophancy | true | dm_multi | Proved | 147 | 0.00 (-4) → 0.79 → 0.00 (+4) | +0.000 (p=1.000) | 0.45 / 0.45 |
| herald | sycophancy | neg | dm_multi | Proved | 150 | 0.00 (-4) → 0.29 → 0.00 (+4) | +0.000 (p=1.000) | 0.20 / 0.20 |
| herald | sycophancy | true | dm_multi | Proved | 148 | 0.00 (-4) → 0.48 → 0.00 (+4) | +0.000 (p=1.000) | 0.45 / 0.45 |
| kimina | proved | neg | dm_multi | Proved | 150 | 0.07 (-4) → 0.23 → 0.10 (+4) | +0.027 (p=0.523) | 0.20 / 0.20 |
| kimina | proved | true | dm_multi | Proved | 150 | 0.29 (-4) → 0.60 → 0.22 (+4) | -0.073 (p=0.144) | 0.50 / 0.50 |
| kimina | sycophancy | neg | dm_multi | Proved | 150 | 0.25 (-4) → 0.33 → 0.32 (+4) | +0.067 (p=0.081) | 0.39 / 0.39 |
| kimina | sycophancy | true | dm_multi | Proved | 149 | 0.50 (-4) → 0.62 → 0.57 (+4) | +0.074 (p=0.045) | 0.65 / 0.65 |
| stepfun | sycophancy | neg | dm_multi | Proved | 150 | 0.30 (-4) → 0.37 → 0.38 (+4) | +0.080 (p=0.066) | 0.40 / 0.40 |
| stepfun | sycophancy | true | dm_multi | Proved | 146 | 0.58 (-4) → 0.68 → 0.65 (+4) | +0.068 (p=0.107) | 0.73 / 0.73 |

**`cot_only`** (5696 statements; Proved outcome only, all directions):

| model | task | steps | direction | outcome | n | rate at min α → 0 → max α | effect over range (perm. p) | random (min α / max α) |
|---|---|---|---|---|---|---|---|---|
| goedel | proved | neg | dm_L24 | Proved | 150 | 0.23 (-8) → 0.33 → 0.17 (+8) | -0.053 (p=0.181) | 0.26 / 0.26 |
| goedel | proved | true | dm_L24 | Proved | 150 | 0.65 (-8) → 0.77 → 0.61 (+8) | -0.047 (p=0.324) | 0.65 / 0.65 |
| goedel | sycophancy | neg | dm_L24 | Proved | 150 | 0.38 (-8) → 0.38 → 0.42 (+8) | +0.040 (p=0.427) | 0.32 / 0.32 |
| goedel | sycophancy | true | dm_L24 | Proved | 147 | 0.80 (-8) → 0.79 → 0.74 (+8) | -0.061 (p=0.093) | 0.78 / 0.78 |
| herald | sycophancy | neg | dm_L16 | Proved | 150 | 0.22 (-8) → 0.29 → 0.23 (+8) | +0.007 (p=0.949) | 0.24 / 0.24 |
| herald | sycophancy | true | dm_L16 | Proved | 148 | 0.37 (-8) → 0.48 → 0.54 (+8) | **+0.169 (p=0.000)** | 0.53 / 0.53 |
| kimina | proved | neg | dm_L28 | Proved | 150 | 0.15 (-8) → 0.23 → 0.27 (+8) | **+0.120 (p=0.002)** | 0.17 / 0.17 |
| kimina | proved | true | dm_L28 | Proved | 150 | 0.37 (-8) → 0.60 → 0.61 (+8) | **+0.240 (p=0.000)** | 0.55 / 0.55 |
| kimina | sycophancy | neg | dm_L28 | Proved | 150 | 0.34 (-8) → 0.33 → 0.32 (+8) | -0.020 (p=0.550) | 0.41 / 0.41 |
| kimina | sycophancy | true | dm_L28 | Proved | 149 | 0.60 (-8) → 0.62 → 0.58 (+8) | -0.020 (p=0.493) | 0.62 / 0.62 |
| stepfun | sycophancy | neg | dm_L28 | Proved | 150 | 0.37 (-8) → 0.37 → 0.37 (+8) | +0.007 (p=0.939) | 0.37 / 0.37 |
| stepfun | sycophancy | true | dm_L28 | Proved | 146 | 0.72 (-8) → 0.68 → 0.67 (+8) | -0.048 (p=0.255) | 0.66 / 0.66 |

**`heads_sigma`** (5577 statements; Proved outcome only, all directions):

| model | task | steps | direction | outcome | n | rate at min α → 0 → max α | effect over range (perm. p) | random (min α / max α) |
|---|---|---|---|---|---|---|---|---|
| goedel | proved | neg | dm_hsig | Proved | 150 | 0.18 (-15) → 0.33 → 0.18 (+15) | +0.000 (p=1.000) | 0.21 / 0.21 |
| goedel | proved | true | dm_hsig | Proved | 150 | 0.45 (-15) → 0.77 → 0.55 (+15) | +0.107 (p=0.049) | 0.56 / 0.56 |
| goedel | sycophancy | neg | dm_hsig | Proved | 150 | 0.23 (-15) → 0.38 → 0.25 (+15) | +0.020 (p=0.692) | 0.23 / 0.23 |
| goedel | sycophancy | true | dm_hsig | Proved | 147 | 0.48 (-15) → 0.79 → 0.63 (+15) | **+0.156 (p=0.002)** | 0.50 / 0.50 |
| herald | sycophancy | neg | dm_hsig | Proved | 150 | 0.01 (-15) → 0.29 → 0.29 (+15) | **+0.280 (p=0.000)** | 0.11 / 0.11 |
| herald | sycophancy | true | dm_hsig | Proved | 148 | 0.01 (-15) → 0.48 → 0.37 (+15) | **+0.358 (p=0.000)** | 0.24 / 0.24 |
| kimina | proved | neg | dm_hsig | Proved | 150 | 0.20 (-15) → 0.23 → 0.20 (+15) | +0.000 (p=1.000) | 0.19 / 0.19 |
| kimina | proved | true | dm_hsig | Proved | 150 | 0.52 (-15) → 0.60 → 0.42 (+15) | -0.100 (p=0.011) | 0.37 / 0.37 |
| kimina | sycophancy | neg | dm_hsig | Proved | 150 | 0.31 (-15) → 0.33 → 0.31 (+15) | +0.000 (p=1.000) | 0.33 / 0.33 |
| kimina | sycophancy | true | dm_hsig | Proved | 149 | 0.50 (-15) → 0.62 → 0.62 (+15) | **+0.121 (p=0.005)** | 0.61 / 0.61 |
| stepfun | sycophancy | neg | dm_hsig | Proved | 150 | 0.30 (-15) → 0.37 → 0.37 (+15) | +0.073 (p=0.068) | 0.34 / 0.34 |
| stepfun | sycophancy | true | dm_hsig | Proved | 146 | 0.58 (-15) → 0.68 → 0.65 (+15) | +0.068 (p=0.120) | 0.70 / 0.70 |

**`ablate`** (5039 statements; Proved outcome only, all directions):

| model | task | steps | direction | outcome | n | rate at min α → 0 → max α | effect over range (perm. p) | random (min α / max α) |
|---|---|---|---|---|---|---|---|---|
| goedel | proved | neg | project_mean | Proved | 150 | 0.33 (+0) → 0.33 → 0.31 (+1) | -0.027 (p=0.572) | – |
| goedel | proved | neg | project_neg_mean | Proved | 150 | 0.33 (+0) → 0.33 → 0.29 (+1) | -0.040 (p=0.345) | – |
| goedel | proved | neg | project_pos_mean | Proved | 150 | 0.33 (+0) → 0.33 → 0.31 (+1) | -0.020 (p=0.678) | – |
| goedel | proved | neg | project_random | Proved | 150 | 0.33 (+0) → 0.33 → 0.29 (+1) | -0.040 (p=0.286) | – |
| goedel | proved | true | project_mean | Proved | 150 | 0.77 (+0) → 0.77 → 0.77 (+1) | -0.007 (p=1.000) | – |
| goedel | proved | true | project_neg_mean | Proved | 150 | 0.77 (+0) → 0.77 → 0.77 (+1) | +0.000 (p=1.000) | – |
| goedel | proved | true | project_pos_mean | Proved | 150 | 0.77 (+0) → 0.77 → 0.81 (+1) | +0.033 (p=0.383) | – |
| goedel | proved | true | project_random | Proved | 150 | 0.77 (+0) → 0.77 → 0.79 (+1) | +0.013 (p=0.824) | – |
| goedel | sycophancy | neg | project_mean | Proved | 150 | 0.38 (+0) → 0.38 → 0.39 (+1) | +0.007 (p=1.000) | – |
| goedel | sycophancy | neg | project_neg_mean | Proved | 150 | 0.38 (+0) → 0.38 → 0.41 (+1) | +0.027 (p=0.636) | – |
| goedel | sycophancy | neg | project_pos_mean | Proved | 150 | 0.38 (+0) → 0.38 → 0.35 (+1) | -0.033 (p=0.473) | – |
| goedel | sycophancy | neg | project_random | Proved | 150 | 0.38 (+0) → 0.38 → 0.36 (+1) | -0.020 (p=0.664) | – |
| goedel | sycophancy | true | project_mean | Proved | 147 | 0.79 (+0) → 0.79 → 0.79 (+1) | +0.000 (p=1.000) | – |
| goedel | sycophancy | true | project_neg_mean | Proved | 147 | 0.79 (+0) → 0.79 → 0.84 (+1) | +0.048 (p=0.189) | – |
| goedel | sycophancy | true | project_pos_mean | Proved | 147 | 0.79 (+0) → 0.79 → 0.80 (+1) | +0.014 (p=0.815) | – |
| goedel | sycophancy | true | project_random | Proved | 147 | 0.79 (+0) → 0.79 → 0.82 (+1) | +0.027 (p=0.388) | – |
| herald | sycophancy | neg | project_mean | Proved | 150 | 0.29 (+0) → 0.29 → 0.26 (+1) | -0.033 (p=0.442) | – |
| herald | sycophancy | neg | project_neg_mean | Proved | 150 | 0.29 (+0) → 0.29 → 0.28 (+1) | -0.013 (p=0.845) | – |
| herald | sycophancy | neg | project_pos_mean | Proved | 150 | 0.29 (+0) → 0.29 → 0.25 (+1) | -0.047 (p=0.248) | – |
| herald | sycophancy | neg | project_random | Proved | 150 | 0.29 (+0) → 0.29 → 0.27 (+1) | -0.020 (p=0.607) | – |
| herald | sycophancy | true | project_mean | Proved | 148 | 0.48 (+0) → 0.48 → 0.47 (+1) | -0.007 (p=1.000) | – |
| herald | sycophancy | true | project_neg_mean | Proved | 148 | 0.48 (+0) → 0.48 → 0.49 (+1) | +0.014 (p=0.815) | – |
| herald | sycophancy | true | project_pos_mean | Proved | 148 | 0.48 (+0) → 0.48 → 0.48 (+1) | +0.000 (p=1.000) | – |
| herald | sycophancy | true | project_random | Proved | 148 | 0.48 (+0) → 0.48 → 0.46 (+1) | -0.020 (p=0.549) | – |
| kimina | proved | neg | project_mean | Proved | 150 | 0.23 (+0) → 0.23 → 0.22 (+1) | -0.013 (p=0.791) | – |
| kimina | proved | neg | project_neg_mean | Proved | 150 | 0.23 (+0) → 0.23 → 0.23 (+1) | -0.007 (p=1.000) | – |
| kimina | proved | neg | project_pos_mean | Proved | 150 | 0.23 (+0) → 0.23 → 0.21 (+1) | -0.027 (p=0.454) | – |
| kimina | proved | neg | project_random | Proved | 150 | 0.23 (+0) → 0.23 → 0.23 (+1) | +0.000 (p=1.000) | – |
| kimina | proved | true | project_mean | Proved | 150 | 0.60 (+0) → 0.60 → 0.57 (+1) | -0.033 (p=0.332) | – |
| kimina | proved | true | project_neg_mean | Proved | 150 | 0.60 (+0) → 0.60 → 0.55 (+1) | -0.047 (p=0.167) | – |
| kimina | proved | true | project_pos_mean | Proved | 150 | 0.60 (+0) → 0.60 → 0.57 (+1) | -0.033 (p=0.332) | – |
| kimina | proved | true | project_random | Proved | 150 | 0.60 (+0) → 0.60 → 0.60 (+1) | +0.000 (p=1.000) | – |
| kimina | sycophancy | neg | project_mean | Proved | 150 | 0.33 (+0) → 0.33 → 0.33 (+1) | +0.000 (p=1.000) | – |
| kimina | sycophancy | neg | project_neg_mean | Proved | 150 | 0.33 (+0) → 0.33 → 0.33 (+1) | -0.007 (p=1.000) | – |
| kimina | sycophancy | neg | project_pos_mean | Proved | 150 | 0.33 (+0) → 0.33 → 0.31 (+1) | -0.027 (p=0.523) | – |
| kimina | sycophancy | neg | project_random | Proved | 150 | 0.33 (+0) → 0.33 → 0.35 (+1) | +0.013 (p=0.625) | – |
| kimina | sycophancy | true | project_mean | Proved | 149 | 0.62 (+0) → 0.62 → 0.54 (+1) | -0.081 (p=0.012) | – |
| kimina | sycophancy | true | project_neg_mean | Proved | 149 | 0.62 (+0) → 0.62 → 0.56 (+1) | -0.060 (p=0.049) | – |
| kimina | sycophancy | true | project_pos_mean | Proved | 149 | 0.62 (+0) → 0.62 → 0.56 (+1) | -0.054 (p=0.077) | – |
| kimina | sycophancy | true | project_random | Proved | 149 | 0.62 (+0) → 0.62 → 0.62 (+1) | +0.007 (p=1.000) | – |
| stepfun | sycophancy | neg | project_mean | Proved | 150 | 0.37 (+0) → 0.37 → 0.25 (+1) | **-0.120 (p=0.006)** | – |
| stepfun | sycophancy | neg | project_neg_mean | Proved | 150 | 0.37 (+0) → 0.37 → 0.21 (+1) | **-0.167 (p=0.000)** | – |
| stepfun | sycophancy | neg | project_pos_mean | Proved | 150 | 0.37 (+0) → 0.37 → 0.29 (+1) | -0.087 (p=0.060) | – |
| stepfun | sycophancy | neg | project_random | Proved | 150 | 0.37 (+0) → 0.37 → 0.43 (+1) | +0.053 (p=0.134) | – |
| stepfun | sycophancy | true | project_mean | Proved | 146 | 0.68 (+0) → 0.68 → 0.58 (+1) | -0.110 (p=0.020) | – |
| stepfun | sycophancy | true | project_neg_mean | Proved | 146 | 0.68 (+0) → 0.68 → 0.48 (+1) | **-0.205 (p=0.000)** | – |
| stepfun | sycophancy | true | project_pos_mean | Proved | 146 | 0.68 (+0) → 0.68 → 0.58 (+1) | -0.110 (p=0.026) | – |
| stepfun | sycophancy | true | project_random | Proved | 146 | 0.68 (+0) → 0.68 → 0.70 (+1) | +0.014 (p=0.804) | – |

- **files:** `prove_<variant>/results.json`, `<model>/steer_<variant>/<task>/generations.jsonl`
- **code:** `run/variants.py::run_steer_variant`; variant specs in `configs/default.yaml::variants`
- **recompute:** `steer-variant --variant <v>` → `prove-variant-gen --variant <v>` → `prove-variant-verify --variant <v>`

### 8.5 One-number summary: does any intervention lower UFLB on the sycophancy rows?


| intervention family | goedel | stepfun | kimina | herald |
|---|---|---|---|---|
| residual (steer) | -0.007 / -0.002 | -0.030 / -0.025 | -0.015 / +0.008 | -0.048 / -0.042 |
| heads top-16 | -0.007 / +0.006 | -0.044 / -0.024 | -0.015 / -0.008 | -0.075 / +0.049 |
| layer_sweep | -0.014 / -0.027 | -0.057 / -0.074 | -0.022 / +0.005 | -0.091 / -0.085 |
| multi_layer | +0.726 / +0.356 | -0.057 / -0.064 | +0.082 / -0.002 | +0.549 / +0.032 |
| cot_only | +0.036 / -0.027 | -0.054 / -0.041 | -0.012 / +0.035 | -0.071 / -0.008 |
| heads_sigma | +0.206 / +0.273 | -0.034 / -0.051 | -0.002 / +0.105 | +0.005 / +0.202 |
| ablate | -0.037 / -0.007 | +0.089 / -0.024 | +0.028 / +0.002 | -0.048 / -0.018 |

Cells: largest UFLB *decrease* (ΔUFLB, negative = more faithful) over the family's non-random conditions / the same for its random control (cells with ≥ 100 steps; UFLB from the per-cell Proved/Refuted/TC_Fail/AF_Fail rates in `results.json::table`).


- **files:** `prove*/results.json::table`
- **code:** this script, `build_background.py` (same arithmetic as the README section)

_Interpretation (human):_ no family lowers UFLB beyond its random control at a
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


## 9. Robustness to the prover (Goedel-Prover-V2-8B relabel)

**TL;DR.** 1,500 held-out benchmark statements per model re-proved with a
different prover: verdict agreement 0.72–0.78 (0.88–0.93 on proved-vs-not);
probe AUROC against the new labels within ≈ 0.03 of the benchmark labels on
the same rows for every model and task.


| model | task | rows / verdict agreement | layer | n bench / new | fixed dir: bench labels | fixed dir: Goedel-Prover labels | CV: bench | CV: new | fixed dir on agreed rows (n) |
|---|---|---|---|---|---|---|---|---|---|
| goedel | proved | 1500 / agree 0.72 (0.88 proved-vs-not) | 24 | 1500 / 1500 | 0.722 | 0.711 | 0.673 | 0.643 | 0.734 (1367) |
| goedel | proved_strict | 1500 / agree 0.72 (0.88 proved-vs-not) | 36 | 968 / 716 | 0.834 | 0.805 | 0.659 | 0.657 | 0.815 (684) |
| goedel | failing | 1500 / agree 0.72 (0.88 proved-vs-not) | 36 | 1500 / 1500 | 0.693 | 0.572 | 0.658 | 0.650 | 0.684 (1422) |
| goedel | sycophancy | 1500 / agree 0.72 (0.88 proved-vs-not) | 24 | 732 / 601 | 0.848 | 0.869 | 0.671 | 0.706 | 0.870 (579) |
| kimina | proved | 1500 / agree 0.74 (0.90 proved-vs-not) | 28 | 1500 / 1500 | 0.720 | 0.708 | 0.609 | 0.625 | 0.731 (1404) |
| kimina | proved_strict | 1500 / agree 0.74 (0.90 proved-vs-not) | 28 | 583 / 444 | 0.817 | 0.790 | 0.616 | 0.609 | 0.800 (397) |
| kimina | failing | 1500 / agree 0.74 (0.90 proved-vs-not) | 14 | 1500 / 1500 | 0.687 | 0.673 | 0.663 | 0.644 | 0.686 (1412) |
| kimina | sycophancy | 1500 / agree 0.74 (0.90 proved-vs-not) | 28 | 454 / 378 | 0.888 | 0.833 | 0.574 | 0.613 | 0.883 (340) |
| herald | proved | 1500 / agree 0.78 (0.93 proved-vs-not) | 26 | 1500 / 1500 | 0.666 | 0.653 | 0.581 | 0.590 | 0.669 (1428) |
| herald | proved_strict | 1500 / agree 0.78 (0.93 proved-vs-not) | 16 | 623 / 498 | 0.874 | 0.839 | 0.662 | 0.636 | 0.873 (458) |
| herald | failing | 1500 / agree 0.78 (0.93 proved-vs-not) | 2 | 1500 / 1500 | 0.638 | 0.626 | 0.588 | 0.605 | 0.643 (1415) |
| herald | sycophancy | 1500 / agree 0.78 (0.93 proved-vs-not) | 16 | 478 / 432 | 0.900 | 0.893 | 0.640 | 0.656 | 0.910 (400) |
| stepfun | proved | 1500 / agree 0.74 (0.90 proved-vs-not) | 28 | 1500 / 1500 | 0.711 | 0.724 | 0.643 | 0.638 | 0.736 (1385) |
| stepfun | proved_strict | 1500 / agree 0.74 (0.90 proved-vs-not) | 28 | 728 / 573 | 0.826 | 0.827 | 0.630 | 0.636 | 0.832 (522) |
| stepfun | failing | 1500 / agree 0.74 (0.90 proved-vs-not) | 24 | 1500 / 1500 | 0.762 | 0.703 | 0.729 | 0.675 | 0.745 (1430) |
| stepfun | sycophancy | 1500 / agree 0.74 (0.90 proved-vs-not) | 28 | 579 / 493 | 0.897 | 0.878 | 0.672 | 0.644 | 0.898 (458) |

- **files:** `relabel_goedel/{rows.json,statements.json,candidates.jsonl,verified.json,results.json}`
- **code:** `analysis/relabel.py` (`benchmark_statement`, `run_relabel_select`, `run_relabel_eval`), `run/prove.py::unnegate_full`
- **recompute:** `relabel-select` → `relabel-gen` (GPU, bmuq) → `relabel-verify` → `relabel-eval`
- **note:** fixed-direction AUROCs are higher than on the full test set because this subset is uniformly sampled, not class-balanced; compare bench vs new on identical rows

## 10. Generalization beyond FaithformBench

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


### 10.1 PRM800K phase-2 test, raw human ratings — 1600 rows, 800 true


Benchmark metrics of the four autoformalizers on this data (our prover pass; UFLB = ½FNR + ½FPR + AFFR, lower is better):

| model | TPR | FNR | FPR | AFFR | UFLB | true: Proved / Refuted / Unknown / TC_Fail | false: Proved / Refuted / TC_Fail |
|---|---|---|---|---|---|---|---|
| goedel | 0.61 | 0.09 | 0.45 | 0.07 | **0.34** | 0.61 / 0.09 / 0.22 / 0.08 | 0.45 / 0.20 / 0.05 |
| kimina | 0.34 | 0.19 | 0.23 | 0.22 | **0.42** | 0.34 / 0.19 / 0.26 / 0.21 | 0.23 / 0.26 / 0.22 |
| herald | 0.35 | 0.14 | 0.28 | 0.28 | **0.48** | 0.35 / 0.14 / 0.21 / 0.24 | 0.28 / 0.19 / 0.21 |
| stepfun | 0.41 | 0.12 | 0.28 | 0.20 | **0.40** | 0.41 / 0.12 / 0.26 / 0.20 | 0.28 / 0.22 / 0.16 |

Truth (true vs false step) from the prompt_last state:

| model | FB corruption-detector dir | FB sycophancy dir | FB proved_strict dir | retrained CV (FB layer) | best-layer CV (optimistic) | TF-IDF CV |
|---|---|---|---|---|---|---|
| goedel | 0.662 | 0.590 | 0.682 | 0.726 | 0.743 (L24) | 0.686 |
| kimina | 0.511 | 0.500 | 0.537 | 0.663 | 0.700 (L12) | 0.686 |
| herald | 0.593 | 0.556 | 0.620 | 0.675 | 0.712 (L20) | 0.686 |
| stepfun | 0.589 | 0.555 | 0.592 | 0.719 | 0.741 (L20) | 0.686 |

Verdict tasks on our prover's verdicts:

| model | task | n (pos) | FB direction | retrained CV | TF-IDF CV |
|---|---|---|---|---|---|
| goedel | proved_unp | 800 (490) | – | – | 0.631 |
| goedel | sycophancy | 852 (490) | 0.555 | 0.698 | 0.622 |
| goedel | proved_strict | 649 (490) | 0.718 | 0.763 | 0.621 |
| goedel | failing | 1600 (1489) | 0.646 | 0.749 | 0.664 |
| kimina | proved_unp | 800 (271) | – | – | 0.597 |
| kimina | sycophancy | 448 (271) | 0.534 | 0.687 | 0.597 |
| kimina | proved_strict | 476 (271) | 0.582 | 0.745 | 0.639 |
| kimina | failing | 1600 (1244) | 0.591 | 0.706 | 0.698 |
| herald | proved_unp | 800 (283) | – | – | 0.601 |
| herald | sycophancy | 498 (283) | 0.556 | 0.696 | 0.577 |
| herald | proved_strict | 438 (283) | 0.630 | 0.770 | 0.655 |
| herald | failing | 1600 (1159) | 0.576 | 0.639 | 0.674 |
| stepfun | proved_unp | 800 (322) | – | – | 0.614 |
| stepfun | sycophancy | 539 (322) | 0.552 | 0.654 | 0.665 |
| stepfun | proved_strict | 497 (322) | 0.599 | 0.715 | 0.669 |
| stepfun | failing | 1600 (1268) | 0.655 | 0.683 | 0.730 |

### 10.2 PRM800K restricted to claim-level steps (LLM judge) — 1600 rows, 800 true


Benchmark metrics of the four autoformalizers on this data (our prover pass; UFLB = ½FNR + ½FPR + AFFR, lower is better):

| model | TPR | FNR | FPR | AFFR | UFLB | true: Proved / Refuted / Unknown / TC_Fail | false: Proved / Refuted / TC_Fail |
|---|---|---|---|---|---|---|---|
| goedel | 0.64 | 0.07 | 0.38 | 0.07 | **0.30** | 0.64 / 0.07 / 0.21 / 0.07 | 0.38 / 0.23 / 0.07 |
| kimina | 0.35 | 0.18 | 0.19 | 0.20 | **0.38** | 0.35 / 0.18 / 0.27 / 0.20 | 0.19 / 0.29 / 0.18 |
| herald | 0.36 | 0.14 | 0.22 | 0.27 | **0.45** | 0.36 / 0.14 / 0.21 / 0.25 | 0.22 / 0.26 / 0.20 |
| stepfun | 0.46 | 0.12 | 0.24 | 0.20 | **0.38** | 0.46 / 0.12 / 0.24 / 0.15 | 0.24 / 0.24 / 0.18 |

Truth (true vs false step) from the prompt_last state:

| model | FB corruption-detector dir | FB sycophancy dir | FB proved_strict dir | retrained CV (FB layer) | best-layer CV (optimistic) | TF-IDF CV |
|---|---|---|---|---|---|---|
| goedel | 0.696 | 0.665 | 0.683 | 0.772 | 0.783 (L24) | 0.632 |
| kimina | 0.590 | 0.536 | 0.571 | 0.673 | 0.708 (L16) | 0.632 |
| herald | 0.638 | 0.593 | 0.606 | 0.729 | 0.729 (L16) | 0.632 |
| stepfun | 0.662 | 0.633 | 0.640 | 0.748 | 0.749 (L20) | 0.632 |

Verdict tasks on our prover's verdicts:

| model | task | n (pos) | FB direction | retrained CV | TF-IDF CV |
|---|---|---|---|---|---|
| goedel | proved_unp | 800 (517) | – | – | 0.660 |
| goedel | sycophancy | 818 (517) | 0.632 | 0.732 | 0.635 |
| goedel | proved_strict | 703 (517) | 0.770 | 0.847 | 0.638 |
| goedel | failing | 1600 (1488) | 0.655 | 0.723 | 0.640 |
| kimina | proved_unp | 800 (276) | – | – | 0.659 |
| kimina | sycophancy | 423 (276) | 0.513 | 0.674 | 0.572 |
| kimina | proved_strict | 512 (276) | 0.627 | 0.792 | 0.627 |
| kimina | failing | 1600 (1273) | 0.569 | 0.600 | 0.652 |
| herald | proved_unp | 800 (286) | – | – | 0.617 |
| herald | sycophancy | 460 (286) | 0.601 | 0.685 | 0.556 |
| herald | proved_strict | 500 (286) | 0.640 | 0.761 | 0.612 |
| herald | failing | 1600 (1170) | 0.583 | 0.675 | 0.701 |
| stepfun | proved_unp | 800 (362) | – | – | 0.645 |
| stepfun | sycophancy | 548 (362) | 0.593 | 0.656 | 0.637 |
| stepfun | proved_strict | 557 (362) | 0.642 | 0.827 | 0.581 |
| stepfun | failing | 1600 (1280) | 0.636 | 0.730 | 0.741 |

### 10.3 miniF2F (488) + adversary-prompt corruptions — 974 rows, 488 true


Benchmark metrics of the four autoformalizers on this data (our prover pass; UFLB = ½FNR + ½FPR + AFFR, lower is better):

| model | TPR | FNR | FPR | AFFR | UFLB | true: Proved / Refuted / Unknown / TC_Fail | false: Proved / Refuted / TC_Fail |
|---|---|---|---|---|---|---|---|
| goedel | 0.51 | 0.03 | 0.10 | 0.10 | **0.16** | 0.51 / 0.03 / 0.36 / 0.10 | 0.10 / 0.39 / 0.09 |
| kimina | 0.43 | 0.04 | 0.09 | 0.16 | **0.23** | 0.43 / 0.04 / 0.37 / 0.17 | 0.09 / 0.32 / 0.16 |
| herald | 0.36 | 0.07 | 0.12 | 0.20 | **0.29** | 0.36 / 0.07 / 0.38 / 0.18 | 0.12 / 0.32 / 0.18 |
| stepfun | 0.42 | 0.02 | 0.07 | 0.18 | **0.23** | 0.42 / 0.02 / 0.37 / 0.17 | 0.07 / 0.34 / 0.16 |

Truth (true vs false step) from the prompt_last state:

| model | FB corruption-detector dir | FB sycophancy dir | FB proved_strict dir | retrained CV (FB layer) | best-layer CV (optimistic) | TF-IDF CV |
|---|---|---|---|---|---|---|
| goedel | 0.573 | 0.530 | 0.560 | 0.598 | 0.610 (L24) | 0.554 |
| kimina | 0.516 | 0.513 | 0.504 | 0.601 | 0.624 (L8) | 0.554 |
| herald | 0.535 | 0.546 | 0.527 | 0.599 | 0.614 (L4) | 0.554 |
| stepfun | 0.555 | 0.550 | 0.537 | 0.593 | 0.606 (L24) | 0.554 |

Verdict tasks on our prover's verdicts:

| model | task | n (pos) | FB direction | retrained CV | TF-IDF CV |
|---|---|---|---|---|---|
| goedel | proved_unp | 488 (249) | – | – | 0.755 |
| goedel | sycophancy | 299 (249) | 0.539 | 0.623 | 0.655 |
| goedel | proved_strict | 439 (249) | 0.580 | 0.571 | 0.536 |
| goedel | failing | 974 (881) | 0.664 | 0.877 | 0.814 |
| kimina | proved_unp | 488 (209) | – | – | 0.743 |
| kimina | sycophancy | 251 (209) | 0.436 | 0.614 | 0.583 |
| kimina | proved_strict | 366 (209) | 0.548 | 0.604 | 0.565 |
| kimina | failing | 974 (815) | 0.618 | 0.791 | 0.747 |
| herald | proved_unp | 488 (175) | – | – | 0.734 |
| herald | sycophancy | 234 (175) | 0.681 | 0.706 | 0.654 |
| herald | proved_strict | 332 (175) | 0.463 | 0.606 | 0.563 |
| herald | failing | 974 (783) | 0.590 | 0.645 | 0.698 |
| stepfun | proved_unp | 488 (203) | – | – | 0.771 |
| stepfun | sycophancy | 237 (203) | 0.610 | 0.734 | 0.663 |
| stepfun | proved_strict | 369 (203) | 0.560 | 0.592 | 0.542 |
| stepfun | failing | 974 (796) | 0.524 | 0.768 | 0.745 |

### 10.4 miniF2F + 1,500 Lean-Workbook statements, corrupted — 3958 rows, 1988 true


Benchmark metrics of the four autoformalizers on this data (our prover pass; UFLB = ½FNR + ½FPR + AFFR, lower is better):

| model | TPR | FNR | FPR | AFFR | UFLB | true: Proved / Refuted / Unknown / TC_Fail | false: Proved / Refuted / TC_Fail |
|---|---|---|---|---|---|---|---|
| goedel | 0.35 | 0.04 | 0.14 | 0.08 | **0.17** | 0.35 / 0.04 / 0.54 / 0.07 | 0.14 / 0.33 / 0.08 |
| kimina | 0.28 | 0.06 | 0.10 | 0.14 | **0.22** | 0.28 / 0.06 / 0.51 / 0.14 | 0.10 / 0.32 / 0.14 |
| herald | 0.31 | 0.08 | 0.15 | 0.14 | **0.26** | 0.31 / 0.08 / 0.47 / 0.13 | 0.15 / 0.31 / 0.14 |
| stepfun | 0.27 | 0.06 | 0.10 | 0.18 | **0.26** | 0.27 / 0.06 / 0.49 / 0.14 | 0.10 / 0.33 / 0.14 |

Truth (true vs false step) from the prompt_last state:

| model | FB corruption-detector dir | FB sycophancy dir | FB proved_strict dir | retrained CV (FB layer) | best-layer CV (optimistic) | TF-IDF CV |
|---|---|---|---|---|---|---|
| goedel | 0.574 | 0.552 | 0.572 | 0.665 | 0.689 (L20) | 0.644 |
| kimina | 0.521 | 0.462 | 0.505 | 0.686 | 0.697 (L8) | 0.644 |
| herald | 0.532 | 0.530 | 0.548 | 0.668 | 0.694 (L8) | 0.644 |
| stepfun | 0.537 | 0.521 | 0.593 | 0.706 | 0.706 (L28) | 0.644 |

Verdict tasks on our prover's verdicts:

| model | task | n (pos) | FB direction | retrained CV | TF-IDF CV |
|---|---|---|---|---|---|
| goedel | proved_unp | 1988 (686) | – | – | 0.738 |
| goedel | sycophancy | 954 (686) | 0.597 | 0.695 | 0.664 |
| goedel | proved_strict | 1334 (686) | 0.622 | 0.764 | 0.699 |
| goedel | failing | 3958 (3650) | 0.700 | 0.926 | 0.870 |
| kimina | proved_unp | 1988 (565) | – | – | 0.744 |
| kimina | sycophancy | 753 (565) | 0.523 | 0.689 | 0.617 |
| kimina | proved_strict | 1201 (565) | 0.561 | 0.769 | 0.724 |
| kimina | failing | 3958 (3388) | 0.581 | 0.796 | 0.810 |
| herald | proved_unp | 1988 (622) | – | – | 0.639 |
| herald | sycophancy | 916 (622) | 0.610 | 0.704 | 0.644 |
| herald | proved_strict | 1235 (622) | 0.533 | 0.757 | 0.686 |
| herald | failing | 3958 (3390) | 0.631 | 0.718 | 0.785 |
| stepfun | proved_unp | 1988 (536) | – | – | 0.744 |
| stepfun | sycophancy | 742 (536) | 0.609 | 0.679 | 0.633 |
| stepfun | proved_strict | 1184 (536) | 0.676 | 0.772 | 0.715 |
| stepfun | failing | 3958 (3257) | 0.605 | 0.761 | 0.767 |

- **files:** `ext_<name>/{steps.parquet,eval.json,verdicts.json}`, `ext_<name>/<model>/{acts/,generations.jsonl}`, `ext_<name>/prove/results.json`
- **inputs:** `/scratch/users/iacopog/leansteer/external/prm800k/{phase2_test.jsonl,judged.jsonl}`, `external/minif2f/perturbed.jsonl`, `external/minif2f_ext/{statements.jsonl,perturbed.jsonl}`
- **code:** `data/external.py` (loaders, filters, balanced problem-grouped sampling), `run/external.py`, `analysis/external_eval.py`; judge `scripts/judge_prm800k.py`; corruption `scripts/perturb_minif2f.py` (the benchmark's adversary prompt, Gemma-3-12B-it)
- **judge statistics:** of 5,067 steps rated −1: 1,687 FALSE_CLAIM, 3,310 TRUE_CLAIM, 70 NO_CLAIM; of 5,000 rated +1: 4,494 TRUE_CLAIM, 337 FALSE_CLAIM, 169 NO_CLAIM (gpuq log of `scripts/ext2_jobs.sh prm`)
- **recompute:** `scripts/external_job.sh <name> <gpu_util>` runs ext-collect → ext-generate → ext-eval → ext-prove-gen → ext-prove-verify → ext-eval

## 11. Conclusions, caveats, open ends

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

**Caveats to state in the report.** Lean type-check proxy uses Mathlib v4.33.1
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
