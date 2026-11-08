"""External step datasets for the generalization check (point 6).

``prm800k``  PRM800K phase-2 test set (openai/prm800k): every rated completion
             of every step (chosen and alternative), rating +1 = true step,
             -1 = false step, restricted to mathematical statements (contains a
             digit, an operator, LaTeX or a fraction; not an "# Answer" line;
             15-300 characters).  Balanced sample of ``n_per_class`` per class,
             whole problems kept together.
``minif2f``  miniF2F (cat-searcher/minif2f-lean4, valid + test): the informal
             statement (true) and its LLM corruption produced with the
             benchmark's own adversary prompt (scripts/perturb_minif2f.py).

Both yield the same frame as the benchmark loader: uid, problem_id, node_id,
step_content, is_perturbed (= false step), subset, plus ``truth`` (1 true / 0
false).  Verdicts come later from our own prover pass (``ext-prove-*``).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from leansteer import config

EXT_ROOT = Path(config.SCRATCH_ROOT) / "external"
_MATHY = re.compile(r"[=<>]|\$|\\frac|\d")


def _pid(problem: str) -> str:
    return "prm-" + hashlib.sha1(problem.strip().encode()).hexdigest()[:10]


def load_prm800k(n_per_class: int = 800, seed: int = 0, path: Optional[Path] = None, per_problem: int = 4) -> pd.DataFrame:
    path = path or EXT_ROOT / "prm800k" / "phase2_test.jsonl"
    rows = []
    seen = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        pid = _pid(r["question"]["problem"])
        for i, s in enumerate(r["label"]["steps"]):
            for j, c in enumerate(s["completions"]):
                t = (c.get("text") or "").strip(); rt = c.get("rating")
                if rt not in (1, -1) or t.startswith("# Answer") or not (15 <= len(t) <= 300) or not _MATHY.search(t):
                    continue
                key = (pid, t)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"problem_id": pid, "node_id": f"step_{i}_{j}", "step_content": t, "is_perturbed": rt == -1,
                             "truth": int(rt == 1), "subset": "prm800k", "problem": r["question"]["problem"]})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    # problem-grouped balanced sample: shuffle problems, take rows until both classes are filled
    pids = np.array(df.problem_id.unique()); rng.shuffle(pids)
    keep, n_true, n_false = [], 0, 0
    for p in pids:   # at most ``per_problem`` rows per class and problem, so the sample spans many problems
        g = df[df.problem_id == p]
        t, f = g[g.truth == 1].head(per_problem), g[g.truth == 0].head(per_problem)
        take_t = t.head(max(0, min(len(t), n_per_class - n_true))); take_f = f.head(max(0, min(len(f), n_per_class - n_false)))
        keep += [take_t, take_f]; n_true += len(take_t); n_false += len(take_f)
        if n_true >= n_per_class and n_false >= n_per_class:
            break
    out = pd.concat(keep).reset_index(drop=True)
    out["uid"] = [f"prm|{p}|{n}|{int(x)}" for p, n, x in zip(out.problem_id, out.node_id, out.is_perturbed)]
    return out


def load_prm800k_claims(n_per_class: int = 800, seed: int = 0, path: Optional[Path] = None, per_problem: int = 4) -> pd.DataFrame:
    """PRM800K steps whose rating agrees with an LLM judge's claim verdict
    (scripts/judge_prm800k.py): true = rated +1 and judged TRUE_CLAIM, false =
    rated -1 and judged FALSE_CLAIM.  Steps the judge calls NO_CLAIM (strategy,
    restatement) are excluded, so a false step here is a false proposition."""
    path = path or EXT_ROOT / "prm800k" / "judged.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if (r["rating"], r["judgement"]) not in ((1, "TRUE_CLAIM"), (-1, "FALSE_CLAIM")):
            continue
        rows.append({"problem_id": r["problem_id"], "node_id": r["node_id"], "step_content": r["text"], "is_perturbed": r["rating"] == -1,
                     "truth": int(r["rating"] == 1), "subset": "prm800k_claims", "problem": r["problem"]})
    df = pd.DataFrame(rows)
    return _balanced_by_problem(df, n_per_class, seed, per_problem, "prmc")


def _balanced_by_problem(df: pd.DataFrame, n_per_class: int, seed: int, per_problem: int, tag: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pids = np.array(df.problem_id.unique()); rng.shuffle(pids)
    keep, n_true, n_false = [], 0, 0
    for p in pids:
        g = df[df.problem_id == p]
        t, f = g[g.truth == 1].head(per_problem), g[g.truth == 0].head(per_problem)
        take_t = t.head(max(0, min(len(t), n_per_class - n_true))); take_f = f.head(max(0, min(len(f), n_per_class - n_false)))
        keep += [take_t, take_f]; n_true += len(take_t); n_false += len(take_f)
        if n_true >= n_per_class and n_false >= n_per_class:
            break
    out = pd.concat(keep).reset_index(drop=True)
    out["uid"] = [f"{tag}|{p}|{n}|{int(x)}" for p, n, x in zip(out.problem_id, out.node_id, out.is_perturbed)]
    return out


def load_minif2f(path: Optional[Path] = None, tag: str = "m2f") -> pd.DataFrame:
    path = path or EXT_ROOT / "minif2f" / "perturbed.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows.append({"problem_id": r["id"], "node_id": "stmt", "step_content": r["informal"], "is_perturbed": False, "truth": 1,
                     "subset": f"minif2f-{r['split']}", "problem": r["informal"]})
        if r.get("perturbed"):
            rows.append({"problem_id": r["id"], "node_id": "stmt", "step_content": r["perturbed"], "is_perturbed": True, "truth": 0,
                         "subset": f"minif2f-{r['split']}", "problem": r["informal"]})
    out = pd.DataFrame(rows)
    out["uid"] = [f"{tag}|{p}|{n}|{int(x)}" for p, n, x in zip(out.problem_id, out.node_id, out.is_perturbed)]
    return out


def load_external(name: str, cfg: dict) -> pd.DataFrame:
    ec = cfg["external"]
    if name == "prm800k":
        df = load_prm800k(int(ec["prm800k_per_class"]), int(cfg["analysis"]["seed"]))
    elif name == "prm800k_claims":
        df = load_prm800k_claims(int(ec["prm800k_per_class"]), int(cfg["analysis"]["seed"]))
    elif name == "minif2f":
        df = load_minif2f()
    elif name == "minif2f_ext":   # miniF2F + 1,500 Lean-Workbook statements, all corrupted with the same prompt
        df = load_minif2f(EXT_ROOT / "minif2f_ext" / "perturbed.jsonl", tag="m2fx")
    else:
        raise ValueError(name)
    if cfg.get("limit"):
        df = df.sample(min(int(cfg["limit"]), len(df)), random_state=0).reset_index(drop=True)
    return df
