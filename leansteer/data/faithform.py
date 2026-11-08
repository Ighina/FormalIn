"""FaithformBench-AF loader and the four binary prediction tasks.

``Faithform-AF.csv`` has one row per (problem, step variant, autoformalizer):
the step text, whether it is the true step or its BrokenMath-style LLM
corruption, the autoformalizer, and the prover verdict ``proof_result`` in
{Proved, Refuted, Unknown, TC-Fail, AF-Fail, Declarative}.

Facts about the file that shape this module (verified on the data):

* ``Declarative`` steps were never sent to an autoformalizer (no output, no
  prover run); they carry no model verdict and are dropped.
* For **unperturbed** rows ``autoformalizer_output`` is the raw model
  response; for **perturbed** rows it holds a prover-side artefact instead
  (proof, negation proof, or statement-with-sorry).  The raw response is
  therefore not available for every row, which is why the probes read the
  residual stream *before generation* (prompt-side anchors).
* 30 "perturbed" steps are textually identical to the true step; they are
  dropped (same input, contradictory labels).
* Every perturbed node has an unperturbed counterpart with the same
  ``(problem_id, node_id)``; that pairing is what the sycophancy analysis uses.

Tasks (positives / negatives), each built per autoformalizer:

    proved         unperturbed & Proved / everything else (balanced by sampling)
    proved_strict  unperturbed & Proved / perturbed & Refuted
    failing        verdict not in {AF_Fail, TC_Fail} / verdict in {AF_Fail, TC_Fail}
    sycophancy     unperturbed & Proved / perturbed & Proved
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from leansteer import config

AF_MODELS = ("goedel", "kimina", "herald", "stepfun")

PROVED, REFUTED, UNKNOWN, TC_FAIL, AF_FAIL, DECLARATIVE = (
    "Proved", "Refuted", "Unknown", "TC_Fail", "AF_Fail", "Declarative")
VERDICTS = (PROVED, REFUTED, UNKNOWN, TC_FAIL, AF_FAIL)
FAILING = (AF_FAIL, TC_FAIL)

_VERDICT_MAP = {
    "proved": PROVED, "refuted": REFUTED, "unknown": UNKNOWN,
    "tc-fail": TC_FAIL, "tc_fail": TC_FAIL, "af-fail": AF_FAIL, "af_fail": AF_FAIL,
    "declarative": DECLARATIVE,
}


def normalise_verdict(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return _VERDICT_MAP.get(str(value).strip().lower())


def make_uid(problem_id: str, node_id: str, is_perturbed: bool) -> str:
    return f"{problem_id}|{node_id}|{int(bool(is_perturbed))}"


# --------------------------------------------------------------------------- statements
THEOREM_SORRY = re.compile(r"(^theorem[\s\S]*?sorry)", re.MULTILINE | re.IGNORECASE)
THEOREM_BY = re.compile(r"(^theorem[\s\S]*?:=\s*by)", re.MULTILINE)


def extract_statement(text: Optional[str]) -> Optional[str]:
    """Best-effort ``theorem ... := by sorry`` from either a raw autoformalizer
    response (last fenced theorem ending in ``sorry``) or a prover proof
    (signature up to ``:= by``).  Negation proofs (``theorem not_...``) wrap the
    original statement in ``¬ (∀ ...)`` and are not unwrapped; ``None``."""
    if not isinstance(text, str) or not text.strip():
        return None
    m = THEOREM_SORRY.findall(text)
    if m:
        return m[-1].strip()
    if text.lstrip().startswith("theorem not_"):
        return None
    m = THEOREM_BY.findall(text)
    if m:
        return (m[0].rstrip() + " sorry").strip()
    return None


# --------------------------------------------------------------------------- loading
def load_af(root: Path | str = config.FAITHFORM_ROOT, models: Sequence[str] = AF_MODELS,
            drop_declarative: bool = True, with_outputs: bool = False) -> pd.DataFrame:
    """Long table, one row per (step variant, model), verdicts normalised."""
    root = Path(root)
    cols = ["problem_id", "node_id", "is_perturbed", "type_of_perturbation", "autoformalizer_model",
            "proof_result", "step_content"]
    if with_outputs:
        cols += ["autoformalizer_output", "prover_output"]
    df = pd.read_csv(root / "Faithform-AF.csv", usecols=cols, low_memory=False)
    df = df[df.autoformalizer_model.isin(models)].copy()
    df["is_perturbed"] = df.is_perturbed.astype(bool)
    df["step_content"] = df.step_content.astype(str).str.strip()
    df["verdict"] = df.proof_result.map(normalise_verdict)
    df = df[df.verdict.notna()]
    if drop_declarative:
        df = df[df.verdict != DECLARATIVE]
    df["subset"] = df.problem_id.str.split("-").str[0]
    df["uid"] = [make_uid(p, n, x) for p, n, x in zip(df.problem_id, df.node_id, df.is_perturbed)]

    # drop perturbed variants whose text equals the true step
    truth = df[~df.is_perturbed].drop_duplicates(["problem_id", "node_id"]).set_index(["problem_id", "node_id"]).step_content
    key = pd.MultiIndex.from_arrays([df.problem_id, df.node_id])
    same = df.is_perturbed.to_numpy() & (df.step_content.to_numpy() == truth.reindex(key).to_numpy())
    df = df[~same]
    if with_outputs:
        df["statement"] = df.autoformalizer_output.map(extract_statement)
    df = df.drop(columns=["proof_result"]).reset_index(drop=True)
    return df


def unique_steps(df: pd.DataFrame) -> pd.DataFrame:
    """One row per step variant (``uid``) — the inputs the models read."""
    return (df.drop_duplicates("uid")[["uid", "problem_id", "node_id", "is_perturbed", "step_content", "subset"]]
            .reset_index(drop=True))


# --------------------------------------------------------------------------- tasks
@dataclass(frozen=True)
class Task:
    name: str
    positive: Callable[[pd.DataFrame], np.ndarray]
    negative: Callable[[pd.DataFrame], np.ndarray]
    description: str


def _unp_proved(d):
    return (~d.is_perturbed) & (d.verdict == PROVED)


TASKS: Dict[str, Task] = {
    "proved": Task(
        "proved", _unp_proved, lambda d: ~_unp_proved(d),
        "true step proved vs. everything else (negatives subsampled to balance)"),
    "proved_strict": Task(
        "proved_strict", _unp_proved, lambda d: d.is_perturbed & (d.verdict == REFUTED),
        "true step proved vs. corrupted step refuted"),
    "failing": Task(
        "failing", lambda d: ~d.verdict.isin(FAILING), lambda d: d.verdict.isin(FAILING),
        "formalization survived (Proved/Refuted/Unknown) vs. AF_Fail/TC_Fail"),
    "sycophancy": Task(
        "sycophancy", _unp_proved, lambda d: d.is_perturbed & (d.verdict == PROVED),
        "true step proved vs. corrupted step proved (sycophantic)"),
    # unperturbed-only variants, for the teacher-forced anchors (raw outputs exist only for true steps)
    "proved_unp": Task(
        "proved_unp", _unp_proved, lambda d: (~d.is_perturbed) & (d.verdict != PROVED),
        "true step proved vs. true step with any other verdict"),
    "failing_unp": Task(
        "failing_unp", lambda d: (~d.is_perturbed) & ~d.verdict.isin(FAILING), lambda d: (~d.is_perturbed) & d.verdict.isin(FAILING),
        "true step: formalization survived vs. AF_Fail/TC_Fail"),
    # perturbation controls: is the sycophancy direction a corruption detector?
    "perturbed_proved": Task(
        "perturbed_proved", lambda d: d.is_perturbed & (d.verdict == PROVED), lambda d: d.is_perturbed & (d.verdict == REFUTED),
        "corrupted steps only: proved (sycophantic) vs refuted (faithful) — no perturbation confound"),
    "perturbed_detect": Task(
        "perturbed_detect", lambda d: ~d.is_perturbed, lambda d: d.is_perturbed,
        "true vs corrupted step, any verdict (a pure corruption detector; balanced)"),
}
DEFAULT_BALANCE = ("proved", "perturbed_detect")


def build_task(df_model: pd.DataFrame, task: str, seed: int = 0, balance: bool = False) -> pd.DataFrame:
    """Rows of one autoformalizer with a binary ``label``; ``group`` = problem_id."""
    t = TASKS[task]
    pos = t.positive(df_model).to_numpy()
    neg = t.negative(df_model).to_numpy() & ~pos
    out = df_model[pos | neg].copy()
    out["label"] = pos[pos | neg].astype(int)
    if balance:
        rng = np.random.default_rng(seed)
        n_pos, n_neg = int(out.label.sum()), int((out.label == 0).sum())
        k = min(n_pos, n_neg)
        keep = np.concatenate([
            rng.choice(np.flatnonzero(out.label.to_numpy() == 1), k, replace=False),
            rng.choice(np.flatnonzero(out.label.to_numpy() == 0), k, replace=False)])
        out = out.iloc[np.sort(keep)]
    out["task"] = task
    out["group"] = out.problem_id
    return out.reset_index(drop=True)


def task_summary(df: pd.DataFrame, balance: Sequence[str] = DEFAULT_BALANCE, seed: int = 0) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("autoformalizer_model"):
        for task in TASKS:
            t = build_task(g, task, seed, balance=task in balance)
            rows.append({"model": model, "task": task, "n": len(t), "n_pos": int(t.label.sum()),
                         "n_neg": int((t.label == 0).sum()), "n_problems": t.problem_id.nunique()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- splits
def problem_split(groups: Sequence[str], test_frac: float, seed: int) -> Dict[str, str]:
    """Deterministic train/test assignment of problem ids (all steps of a
    problem, true and corrupted, land on the same side)."""
    ids = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_frac))
    return {g: ("test" if i < n_test else "train") for i, g in enumerate(ids)}
