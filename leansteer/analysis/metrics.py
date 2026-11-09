"""FaithformBench's final metrics (paper v1, arXiv:2608.10916, §4 "Failure statistics").

For a set of true steps (unperturbed) and corrupted steps (perturbed) with
prover verdicts in {Proved, Refuted, Unknown, TC_Fail, AF_Fail}:

    FNR  = #(true steps Refuted)            / #true       "error induction"
    FPR  = #(corrupted steps Proved)        / #corrupted  "silent correction" (sycophancy)
    AFFR = #(steps with AF_Fail or TC_Fail) / #all        the paper's AF-Fail aggregates
                                                          extraction and type-check failures
    UFLB = 1/2 FNR + 1/2 FPR + AFFR                      unfaithfulness lower bound; lower is better

Unknown (the paper's "Inconclusive") counts in no failure event.  When the two
classes are not equally sized the 1/2 weights are kept, as in the paper, which
assumes N true and N corrupted steps; TPR = P(Proved | true) is reported too.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd

AF_FAILURES = ("AF_Fail", "TC_Fail")


def faithfulness_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """``df`` needs ``is_perturbed`` and ``verdict`` columns."""
    tr, co = df[~df.is_perturbed.astype(bool)], df[df.is_perturbed.astype(bool)]
    nan = float("nan")
    fnr = float((tr.verdict == "Refuted").mean()) if len(tr) else nan
    tpr = float((tr.verdict == "Proved").mean()) if len(tr) else nan
    fpr = float((co.verdict == "Proved").mean()) if len(co) else nan
    affr = float(df.verdict.isin(AF_FAILURES).mean()) if len(df) else nan
    uflb = 0.5 * fnr + 0.5 * fpr + affr if not (math.isnan(fnr) or math.isnan(fpr)) else nan
    return {"TPR": tpr, "FNR": fnr, "FPR": fpr, "AFFR": affr, "UFLB": uflb, "n": int(len(df)), "n_true": int(len(tr)), "n_corrupted": int(len(co))}


def uflb_from_rates(true_refuted: float, corrupted_proved: float, true_af: float, corrupted_af: float) -> float:
    """UFLB from per-class rates (equal class weights): AFFR = (true_af + corrupted_af)/2."""
    return 0.5 * true_refuted + 0.5 * corrupted_proved + 0.5 * (true_af + corrupted_af)
