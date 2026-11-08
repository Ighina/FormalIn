import numpy as np
import pandas as pd

from leansteer.data.faithform import (AF_FAIL, PROVED, REFUTED, TC_FAIL, UNKNOWN, build_task, extract_statement,
                                      normalise_verdict, problem_split)


def _frame():
    rows = []
    for p in range(6):
        for n in range(3):
            for pert in (False, True):
                v = [PROVED, REFUTED, UNKNOWN, TC_FAIL, AF_FAIL][(p + n + int(pert)) % 5]
                rows.append(dict(problem_id=f"math-{p}", node_id=f"step_{n}", is_perturbed=pert, verdict=v,
                                 step_content=f"s{p}{n}{int(pert)}", uid=f"math-{p}|step_{n}|{int(pert)}",
                                 autoformalizer_model="goedel", subset="math"))
    return pd.DataFrame(rows)


def test_verdicts():
    assert normalise_verdict("TC-Fail") == TC_FAIL and normalise_verdict("AF-Fail") == AF_FAIL
    assert normalise_verdict("Proved") == PROVED and normalise_verdict(float("nan")) is None


def test_tasks():
    d = _frame()
    t = build_task(d, "proved")
    assert set(t.label) == {0, 1}
    assert ((t.label == 1) == ((~t.is_perturbed) & (t.verdict == PROVED))).all()
    assert len(t) == len(d)
    t = build_task(d, "proved", balance=True)
    assert t.label.sum() == (t.label == 0).sum()
    t = build_task(d, "proved_strict")
    assert (t[t.label == 0].is_perturbed).all() and (t[t.label == 0].verdict == REFUTED).all()
    t = build_task(d, "failing")
    assert set(t[t.label == 0].verdict) <= {AF_FAIL, TC_FAIL} and not (t[t.label == 1].verdict.isin([AF_FAIL, TC_FAIL])).any()
    t = build_task(d, "sycophancy")
    assert (t.verdict == PROVED).all() and (t.label == (~t.is_perturbed).astype(int)).all()


def test_split_is_grouped():
    d = _frame()
    a = problem_split(d.problem_id, 0.3, 0)
    assert set(a.values()) == {"train", "test"}
    assert a == problem_split(d.problem_id, 0.3, 0)
    assert sum(v == "test" for v in a.values()) == 2


def test_extract_statement():
    raw = "<think>blah</think>```lean4\nimport Mathlib\n\ntheorem my_favorite_theorem : 256 = 2^8 := by sorry\n```"
    assert extract_statement(raw) == "theorem my_favorite_theorem : 256 = 2^8 := by sorry"
    proof = "theorem my_favorite_theorem (x : ℕ) (h : x = 1) : x + 1 = 2 := by\n  subst h\n  rfl"
    assert extract_statement(proof) == "theorem my_favorite_theorem (x : ℕ) (h : x = 1) : x + 1 = 2 := by sorry"
    assert extract_statement("theorem not_my_favorite_theorem : ¬ (1 = 2) := by\n  decide") is None
    assert extract_statement("-- Failed to formalize") is None and extract_statement(None) is None
