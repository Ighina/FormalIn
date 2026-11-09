"""Stages ``prove-gen`` and ``prove-verify``: the benchmark's prover protocol
applied to the steered generations.

For every distinct generated statement we build the DeepSeek-Prover-V2 prompt
exactly as the benchmark does (header, docstring with the natural-language
step, statement with ``sorry``, "Complete the following Lean 4 code" wrapper),
sample ``n`` proofs for the statement and for its *full* negation
(``theorem not_<name>_full : ¬ (∀ <binders>, <type>)``), compile every
candidate with Lean + Mathlib, and assign the benchmark verdict:

    TC_FAIL   the statement itself does not elaborate
    PROVED    some candidate proves the statement
    REFUTED   none does, but some candidate proves the negation
    UNKNOWN   neither

``prove-gen`` needs vLLM (run it with the ``bmuq`` env's python on a GPU);
``prove-verify`` needs only Lean and many CPU cores.  The two stages share
``runs/<run>/prove/`` and are resumable.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from leansteer.analysis import lean as leancheck
from leansteer.experiment import Run, write_json

log = logging.getLogger(__name__)

HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n"
LEAN_TEMPLATE = HEADER + "\n/-- {statement}-/\n{lean_code}"
WRAPPER = """Complete the following Lean 4 code:

```lean4
{lean_code}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""
PROVER = "deepseek-ai/DeepSeek-Prover-V2-7B"
PROVERS = {
    # name -> hf id, whether the tokenizer needs the fast-tokenizer copy (transformers >= 5 slow-class problem),
    # and whether the model emits a <think> block before the answer (same "proof plan" prompt for both)
    "deepseek": {"id": "deepseek-ai/DeepSeek-Prover-V2-7B", "fast_copy": True, "thinking": False},
    "goedel": {"id": "Goedel-LM/Goedel-Prover-V2-8B", "fast_copy": False, "thinking": True},
}
_BLOCK = re.compile(r"```lean4?(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN, _CLOSE = "({[⦃", ")}]⦄"


# --------------------------------------------------------------------------- statements
def split_signature(statement: str) -> Tuple[str, List[str], str]:
    """``theorem NAME BINDERS : TYPE := ...`` -> (NAME, [binder groups], TYPE)."""
    s = statement.strip()
    s = re.sub(r":=\s*(by\s*)?sorry\s*$", "", s).rstrip()
    m = re.match(r"theorem\s+(\S+)\s*", s)
    if not m:
        raise ValueError("not a theorem")
    name, rest = m.group(1), s[m.end():]
    binders, i = [], 0
    while i < len(rest) and rest[i] in _OPEN:
        depth, j = 0, i
        while j < len(rest):
            if rest[j] in _OPEN:
                depth += 1
            elif rest[j] in _CLOSE:
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            raise ValueError("unbalanced binder")
        binders.append(rest[i:j + 1])
        i = j + 1
        while i < len(rest) and rest[i].isspace():
            i += 1
    if i >= len(rest) or rest[i] != ":":
        raise ValueError("no type")
    typ = rest[i + 1:].strip()
    if not typ:
        raise ValueError("empty type")
    return name, binders, typ


def negate_full(statement: str) -> Optional[str]:
    try:
        name, binders, typ = split_signature(statement)
    except ValueError:
        return None
    quant = ("∀ " + " ".join(binders) + ", ") if binders else ""
    return f"theorem not_{name}_full : ¬ ({quant}{typ}) := by sorry"


def with_sorry(statement: str) -> str:
    s = re.sub(r":=\s*(by\s*)?sorry\s*$", "", statement.strip()).rstrip()
    return s + " := by sorry"


def prover_prompt(nl: str, statement: str) -> str:
    return WRAPPER.format(lean_code=LEAN_TEMPLATE.format(statement=nl.strip() + " ", lean_code=with_sorry(statement)))


def _norm(s: str) -> str:
    return " ".join(s.split())


def unnegate_full(negation: str) -> Optional[str]:
    """Inverse of ``negate_full`` for the benchmark's stored negation proofs:
    ``theorem not_NAME_full : ¬ (∀ BINDERS, TYPE) := by ...`` -> ``theorem NAME BINDERS : TYPE := by sorry``."""
    m = re.match(r"\s*theorem\s+not_(\S+?)_(?:full|strong)\s*:\s*¬\s*\(", negation)
    if not m:
        return None
    name = m.group(1)
    i = m.end()
    depth, j = 1, i
    while j < len(negation) and depth > 0:
        if negation[j] == "(":
            depth += 1
        elif negation[j] == ")":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    inner = negation[i:j - 1].strip()
    if inner.startswith("∀"):
        inner = inner[1:].strip()
        # binders up to the first top-level comma
        d, k = 0, 0
        while k < len(inner):
            c = inner[k]
            if c in _OPEN:
                d += 1
            elif c in _CLOSE:
                d -= 1
            elif c == "," and d == 0:
                break
            k += 1
        if k >= len(inner):
            return None
        binders, typ = inner[:k].strip(), inner[k + 1:].strip()
        return f"theorem {name} {binders} : {typ} := by sorry"
    return f"theorem {name} : {inner} := by sorry"


def _strip_think(text: str) -> str:
    return text.split("</think>", 1)[1] if "</think>" in text else text


def extract_candidate(response: str, statement: str) -> Optional[str]:
    """Last lean block of the response, stripped of the header, kept only when it
    restates the target signature verbatim (modulo whitespace)."""
    response = _strip_think(response)
    blocks = _BLOCK.findall(response)
    if not blocks:
        return None
    code = blocks[-1].strip()
    code = re.sub(r"^(import .*\n|set_option .*\n|open .*\n|\s*\n)+", "", code)
    try:
        name, binders, typ = split_signature(statement)
    except ValueError:
        return None
    sig = _norm(f"theorem {name} {' '.join(binders)} : {typ}".replace("  ", " "))
    if sig not in _norm(code):
        return None
    if "sorry" in code:
        return None
    return code


# --------------------------------------------------------------------------- selection
def select_generations(run: Run, cfg: dict, name: str = "steer") -> List[dict]:
    pc = cfg["prove"]
    conds = None   # None = every condition in the file
    if name == "steer_heads":   # head steering has its own alpha scale
        hc = cfg["heads"]
        conds = {("dm", float(a)) for a in hc["steer_alphas"]} | {("random", float(a)) for a in hc["steer_random_alphas"]}
    elif name == "steer":
        conds = {(d, float(a)) for d, alphas in pc["conditions"].items() for a in alphas}
    from leansteer.run.steer import current_selection
    rows = []
    for model in cfg["models"]:
        for task in pc["tasks"]:
            p = run.steer_dir(model, task, name) / "generations.jsonl"
            if not p.exists():
                continue
            keep = current_selection(cfg, model, task)
            for line in p.read_text().splitlines():
                g = json.loads(line)
                if g["uid"] not in keep:
                    continue
                if g["alpha"] != 0.0 and g.get("unit_kind") != "dm_norm":
                    continue
                if conds is None or (g["direction"], g["alpha"]) in conds or g["alpha"] == 0.0:
                    g["model"] = model
                    rows.append(g)
    return rows


def ext_rows(run: Run, cfg: dict, name: str) -> List[dict]:
    """Autoformalizations of an external dataset in the generation-record layout."""
    rows = []
    for model in cfg["models"]:
        p = run.root / f"ext_{name}" / model / "generations.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            g = json.loads(line)
            rows.append({**g, "model": model, "task": "ext", "direction": "base", "alpha": 0.0, "role": "eval", "unit_kind": "dm_norm"})
    return rows


def statement_key(statement: str) -> str:
    import hashlib
    return hashlib.sha1(_norm(statement).encode()).hexdigest()[:16]


def unique_statements(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for g in rows:
        if not g.get("statement"):
            continue
        k = statement_key(g["statement"])
        if k not in out:
            out[k] = {"key": k, "statement": g["statement"], "nl": g["step"], "negation": negate_full(g["statement"])}
    return out


# --------------------------------------------------------------------------- generation (vLLM)
def prover_dir(run: Run, which: str, prover: str) -> Path:
    """``prove/`` for the steered set with the default prover (backward compatible),
    ``prove_<prover>/`` otherwise; ``relabel_<prover>/`` for the benchmark statements."""
    if which == "steer":
        d = run.root / ("prove" if prover == "deepseek" else f"prove_{prover}")
    elif which == "steer_heads":
        d = run.root / ("prove_heads" if prover == "deepseek" else f"prove_heads_{prover}")
    elif which.startswith("steer_"):
        d = run.root / (f"prove_{which[6:]}" if prover == "deepseek" else f"prove_{which[6:]}_{prover}")
    elif which.startswith("ext_"):
        d = run.root / which / ("prove" if prover == "deepseek" else f"prove_{prover}")
    else:
        d = run.root / f"relabel_{prover}"
    d.mkdir(exist_ok=True)
    return d


def prover_tokenizer_dir() -> Path:
    """A copy of the prover's tokenizer files declared as ``PreTrainedTokenizerFast``.

    Under transformers >= 5 the repo's ``LlamaTokenizer`` resolves to the slow
    sentencepiece path, which drops spaces on encode and cannot decode (vLLM
    would both mis-tokenize the prompt and return byte-level junk).  Pointing
    vLLM and our own encoding at this directory uses the ``tokenizer.json``
    backend instead; the round-trip is asserted."""
    import shutil
    from huggingface_hub import snapshot_download
    from transformers import PreTrainedTokenizerFast
    from leansteer import config as _cfg
    d = _cfg.SCRATCH_ROOT / "tokenizers" / PROVER.replace("/", "--")
    if not (d / "tokenizer.json").exists():
        src = Path(snapshot_download(PROVER, allow_patterns=["tokenizer*", "*.json"], local_files_only=True))
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "tokenizer.json", d / "tokenizer.json")
        tc = json.loads((src / "tokenizer_config.json").read_text())
        tc["tokenizer_class"] = "PreTrainedTokenizerFast"
        (d / "tokenizer_config.json").write_text(json.dumps(tc))
    tok = PreTrainedTokenizerFast.from_pretrained(str(d))
    probe = "theorem foo (x : ℕ) : x + 1 = 1 + x := by\n  omega"
    ids = tok(probe, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == probe, tok.decode(ids)
    return d


def run_prove_gen(cfg: dict, which: str = "steer") -> Path:
    run = Run(cfg)
    pc = dict(cfg["prove"])
    if which == "relabel":
        pc.update(cfg["relabel"])
    prover = pc.get("prover", "deepseek")
    d = prover_dir(run, which, prover)
    if which.startswith("steer") or which.startswith("ext_"):
        rows = select_generations(run, cfg, which) if which.startswith("steer") else ext_rows(run, cfg, which[4:])
        stmts = unique_statements(rows)
        write_json(d / "statements.json", stmts)
        log.info("%d generations -> %d unique statements", len(rows), len(stmts))
    else:
        stmts = json.loads((d / "statements.json").read_text())
    cand_path = d / "candidates.jsonl"
    done = set()
    if cand_path.exists():
        done = {(r["key"], r["kind"]) for r in map(json.loads, cand_path.read_text().splitlines())}
    jobs = []
    for s in stmts.values():
        if (s["key"], "dir") not in done:
            jobs.append((s["key"], "dir", prover_prompt(s["nl"], s["statement"])))
        if s["negation"] and (s["key"], "neg") not in done:
            jobs.append((s["key"], "neg", prover_prompt("Negation: it is not the case that " + s["nl"], s["negation"])))
    if pc.get("shard"):   # "k/n": this worker takes every n-th pending prompt starting at k-1 (disjoint, resumable)
        k, n = (int(x) for x in str(pc["shard"]).split("/"))
        jobs = jobs[k - 1::n]
    if cfg.get("limit"):
        jobs = jobs[: int(cfg["limit"])]
    log.info("%d prover prompts to run with %s (n=%d samples each, shard %s)", len(jobs), prover, pc["n_samples"], pc.get("shard") or "1/1")
    if not jobs:
        return d
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from vllm import LLM, SamplingParams
    spec = PROVERS[prover]
    if spec["fast_copy"]:
        tok_dir = str(prover_tokenizer_dir())
        tok = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    else:
        tok_dir = spec["id"]
        tok = AutoTokenizer.from_pretrained(spec["id"])
        probe = "theorem foo (x : ℕ) : x + 1 = 1 + x := by\n  omega"
        assert tok.decode(tok(probe, add_special_tokens=False)["input_ids"]) == probe, "tokenizer does not round-trip"
    prompts = [tok.apply_chat_template([{"role": "user", "content": j[2]}], tokenize=False, add_generation_prompt=True) for j in jobs]
    llm = LLM(model=spec["id"], tokenizer=tok_dir, dtype="bfloat16", max_model_len=int(pc["max_model_len"]),
              gpu_memory_utilization=float(pc["gpu_memory_utilization"]), seed=0, enable_prefix_caching=True)
    sp = SamplingParams(n=int(pc["n_samples"]), temperature=0.6, top_p=0.95, max_tokens=int(pc["max_tokens"]), seed=0)
    bs = int(pc["gen_batch"])
    with open(cand_path, "a") as f:
        for i0 in range(0, len(jobs), bs):
            outs = llm.generate(prompts[i0:i0 + bs], sp, use_tqdm=True)
            for (key, kind, _), o in zip(jobs[i0:i0 + bs], outs):
                f.write(json.dumps({"key": key, "kind": kind, "responses": [tok.decode(c.token_ids, skip_special_tokens=True) for c in o.outputs],
                                    "n_tokens": [len(c.token_ids) for c in o.outputs]}) + "\n")
            f.flush()
            log.info("prover: %d/%d prompts done", min(i0 + bs, len(jobs)), len(jobs))
    return d


# --------------------------------------------------------------------------- verification (Lean)
def _verify_chunk(args):
    codes, project_dir, timeout = args
    return leancheck.verify_proofs(codes, project_dir, timeout, batch=len(codes))


def _typecheck_chunk(args):
    stmts, project_dir, timeout = args
    return leancheck.typecheck(stmts, project_dir, timeout, batch=len(stmts))


def _pool_map(fn, chunks, workers):
    if not chunks:
        return []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, chunks))


def run_prove_verify(cfg: dict, which: str = "steer") -> Path:
    run = Run(cfg)
    pc, lc = dict(cfg["prove"]), cfg["lean"]
    if which == "relabel":
        pc.update(cfg["relabel"])
    d = prover_dir(run, which, pc.get("prover", "deepseek"))
    stmts = json.loads((d / "statements.json").read_text())
    cands = [json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines() if l.strip()]
    ver_path = d / "verified.json"
    verified: Dict[str, dict] = json.loads(ver_path.read_text()) if ver_path.exists() else {}
    workers, timeout, batch = int(pc["lean_workers"]), int(lc["timeout"]), int(pc["lean_batch"])

    # 1. statement type-check (cached across the steer stage's typecheck.json files)
    tc_cache: Dict[str, Optional[bool]] = {}
    for model in cfg["models"]:
        for task in pc["tasks"]:
            for name in ("steer", "steer_heads"):
                p = run.steer_dir(model, task, name) / "typecheck.json"
                if p.exists():
                    tc_cache.update(json.loads(p.read_text()))
    # reuse verdicts already established for identical statements by the other steer set
    for q in run.root.glob("prove*/verified.json"):
        if q.exists() and q != ver_path:
            for k, v in json.loads(q.read_text()).items():
                if k in stmts and k not in verified:
                    verified[k] = v
    need = [k for k, s in stmts.items() if k not in verified and s["statement"] not in tc_cache]
    chunks = [( [stmts[k]["statement"] for k in need[i:i + batch]], lc["project_dir"], timeout) for i in range(0, len(need), batch)]
    log.info("type-checking %d statements in %d chunks", len(need), len(chunks))
    for keys, res in zip([need[i:i + batch] for i in range(0, len(need), batch)], _pool_map(_typecheck_chunk, chunks, workers)):
        for k, ok in zip(keys, res):
            tc_cache[stmts[k]["statement"]] = ok

    # 2. proof candidates
    todo: List[Tuple[str, str, int, str]] = []   # (key, kind, sample idx, code)
    n_parsed = n_total = 0
    for c in cands:
        if c["key"] in verified or c["key"] not in stmts:
            continue
        s = stmts[c["key"]]
        target = s["statement"] if c["kind"] == "dir" else s["negation"]
        if tc_cache.get(s["statement"]) is False:
            continue
        for j, resp in enumerate(c["responses"]):
            n_total += 1
            code = extract_candidate(resp, target)
            if code:
                n_parsed += 1
                todo.append((c["key"], c["kind"], j, code))
    log.info("verifying %d proof candidates (%d/%d parsed) with %d workers", len(todo), n_parsed, n_total, workers)
    chunks = [([t[3] for t in todo[i:i + batch]], lc["project_dir"], timeout) for i in range(0, len(todo), batch)]
    results = defaultdict(lambda: {"dir": [], "neg": []})
    for grp, res in zip([todo[i:i + batch] for i in range(0, len(todo), batch)], _pool_map(_verify_chunk, chunks, workers)):
        for (key, kind, j, code), ok in zip(grp, res):
            results[key][kind].append({"sample": j, "pass": ok})
    have = {(c["key"], c["kind"]) for c in cands}
    for k, s in stmts.items():
        if k in verified:
            continue
        if (k, "dir") not in have or (s["negation"] is not None and (k, "neg") not in have):
            continue   # prover has not produced candidates for this statement yet
        tc = tc_cache.get(s["statement"])
        r = results.get(k, {"dir": [], "neg": []})
        proved = any(x["pass"] for x in r["dir"])
        refuted = any(x["pass"] for x in r["neg"])
        verdict = "TC_Fail" if tc is False else "Proved" if proved else "Refuted" if refuted else "Unknown"
        verified[k] = {"verdict": verdict, "typecheck": tc, "n_dir": len(r["dir"]), "n_neg": len(r["neg"]),
                       "dir_pass": sum(x["pass"] is True for x in r["dir"]), "neg_pass": sum(x["pass"] is True for x in r["neg"]),
                       "has_negation": s["negation"] is not None}
    ver_path.write_text(json.dumps(verified))
    return score_prove(cfg, which)


def score_prove(cfg: dict, which: str = "steer") -> Path:
    run = Run(cfg)
    if which == "relabel":
        from leansteer.analysis.relabel import run_relabel_eval
        return run_relabel_eval(cfg)
    if which.startswith("ext_"):
        return score_external(cfg, which[4:])
    d = prover_dir(run, which, cfg["prove"].get("prover", "deepseek"))
    verified = json.loads((d / "verified.json").read_text())
    rows = select_generations(run, cfg, which)
    groups = defaultdict(list)
    for g in rows:
        v = verified.get(statement_key(g["statement"])) if g.get("statement") else None
        verdict = v["verdict"] if v else "AF_Fail"
        groups[(g["model"], g["task"], g["direction"], g["alpha"], g["label"], g["role"])].append(verdict)
    table = []
    for (model, task, dname, alpha, label, role), vs in sorted(groups.items()):
        n = len(vs)
        table.append({"model": model, "task": task, "direction": dname, "alpha": alpha, "label": label, "role": role, "n": n,
                      **{f"{k.lower()}_rate": float(np.mean([v == k for v in vs])) for k in ("Proved", "Refuted", "Unknown", "TC_Fail", "AF_Fail")}})
    out = {"n_statements": len(verified), "table": table, "trends": dose_response(rows, verified),
           "verdicts_overall": {k: sum(1 for v in verified.values() if v["verdict"] == k) for k in ("Proved", "Refuted", "Unknown", "TC_Fail")}}
    write_json(d / "results.json", out)
    return d / "results.json"


def score_external(cfg: dict, name: str) -> Path:
    """Per-model verdicts of the external dataset's autoformalizations and the
    benchmark metrics (TPR, FNR, FPR, AFFR, UFLB) by truth label."""
    run = Run(cfg)
    d = prover_dir(run, f"ext_{name}", cfg["prove"].get("prover", "deepseek"))
    verified = json.loads((d / "verified.json").read_text())
    rows = ext_rows(run, cfg, name)
    verdicts, table = {}, []
    by = defaultdict(list)
    for g in rows:
        v = verified.get(statement_key(g["statement"])) if g.get("statement") else None
        verdict = v["verdict"] if v else "AF_Fail"
        verdicts[f"{g['model']}|{g['uid']}"] = verdict
        by[(g["model"], int(g["truth"]))].append(verdict)
    from leansteer.analysis.metrics import uflb_from_rates
    for model in cfg["models"]:
        t, f = by.get((model, 1), []), by.get((model, 0), [])
        if not t:
            continue
        rate = lambda vs, k: float(np.mean([v == k for v in vs])) if vs else float("nan")
        tpr, fnr = rate(t, "Proved"), rate(t, "Refuted"); fpr = rate(f, "Proved")
        af_t = float(np.mean([v in ("AF_Fail", "TC_Fail") for v in t])); af_f = float(np.mean([v in ("AF_Fail", "TC_Fail") for v in f])) if f else float("nan")
        uflb = uflb_from_rates(fnr, fpr, af_t, af_f) if f else float("nan")
        table.append({"model": model, "n_true": len(t), "n_false": len(f), "TPR": tpr, "FNR": fnr, "FPR": fpr, "AFFR": 0.5 * (af_t + af_f) if f else af_t, "UFLB": uflb,
                      **{f"true_{k.lower()}": rate(t, k) for k in ("Proved", "Refuted", "Unknown", "TC_Fail", "AF_Fail")},
                      **{f"false_{k.lower()}": (rate(f, k) if f else None) for k in ("Proved", "Refuted", "Unknown", "TC_Fail", "AF_Fail")}})
    out = {"dataset": name, "n_statements": len(verified), "table": table}
    write_json(d / "results.json", out)
    write_json(run.root / f"ext_{name}" / "verdicts.json", verdicts)
    return d / "results.json"


def dose_response(rows: List[dict], verified: dict, n_perm: int = 2000, seed: int = 0) -> List[dict]:
    """Per (model, task, label, direction family, outcome): slope of the outcome
    rate on alpha (alpha = 0 included), with a permutation p-value that shuffles
    alpha *within each step*, and the matching random family's rates.  A family
    is a direction name not starting with ``random``; its random control is
    ``random`` + the same suffix (``dm_L12`` -> ``random_L12``, ``dm`` ->
    ``random``).  Single-alpha families (projection ablations) get a paired
    sign test of the outcome against alpha = 0 instead of a slope."""
    rng = np.random.default_rng(seed)
    fam = defaultdict(lambda: defaultdict(dict))   # (model, task, label, direction) -> uid -> alpha -> verdict
    base = defaultdict(dict)                       # (model, task, label) -> uid -> verdict at alpha 0
    for g in rows:
        if g["role"] != "eval":
            continue
        v = verified.get(statement_key(g["statement"])) if g.get("statement") else None
        verdict = v["verdict"] if v else "AF_Fail"
        key = (g["model"], g["task"], g["label"])
        if g["alpha"] == 0.0:
            base[key][g["uid"]] = verdict
        else:
            fam[(*key, g["direction"])][g["uid"]][g["alpha"]] = verdict
    out = []
    for (model, task, label, dname), per_uid in sorted(fam.items()):
        if dname.startswith("random"):
            continue
        suffix = dname[2:] if dname.startswith("dm") else ""
        rkey = (model, task, label, "random" + suffix)
        rnd = fam.get(rkey, {})
        # alphas available for (nearly) every step: a later run may have added steps with a smaller alpha grid
        counts = defaultdict(int)
        for d in per_uid.values():
            for a in d:
                counts[a] += 1
        alphas = sorted(a for a, c in counts.items() if c >= 0.9 * len(per_uid))
        b = base[(model, task, label)]
        uids = [u for u, d in per_uid.items() if all(a in d for a in alphas) and u in b]
        if len(uids) < 20:
            continue
        for outcome in ("Proved", "Refuted", "TC_Fail"):
            if dname.startswith("project"):
                y1 = np.array([per_uid[u][alphas[0]] == outcome for u in uids], dtype=float)
                y0 = np.array([b[u] == outcome for u in uids], dtype=float)
                diff = y1 - y0
                n_pos, n_neg = int((diff > 0).sum()), int((diff < 0).sum())
                from scipy.stats import binomtest
                p = float(binomtest(n_pos, n_pos + n_neg).pvalue) if n_pos + n_neg else 1.0
                out.append({"model": model, "task": task, "label": label, "outcome": outcome, "direction": dname, "n_steps": len(uids),
                            "range_effect": float(diff.mean()), "perm_p": p, "rate_by_alpha": {"0.0": float(y0.mean()), str(alphas[0]): float(y1.mean())},
                            "random_rate": {}})
                continue
            A = np.array([0.0] + alphas, dtype=float)
            Y = np.array([[b[u] == outcome] + [per_uid[u][a] == outcome for a in alphas] for u in uids], dtype=float)
            def slope(M):
                return float(np.polyfit(A, M.mean(0), 1)[0])
            obs = slope(Y)
            perms = np.empty(n_perm)
            for i in range(n_perm):
                P = np.array([rng.permutation(len(A)) for _ in uids])
                perms[i] = slope(np.take_along_axis(Y, P, axis=1))
            p = float((np.abs(perms) >= abs(obs)).mean())
            rates = {str(a): float(Y[:, j].mean()) for j, a in enumerate(A)}
            ralphas = sorted({a for d in rnd.values() for a in d})
            rand = {str(a): float(np.mean([d[a] == outcome for d in rnd.values() if a in d])) for a in ralphas}
            out.append({"model": model, "task": task, "label": label, "outcome": outcome, "direction": dname, "n_steps": len(uids),
                        "slope_per_alpha": obs, "range_effect": obs * (A.max() - A.min()), "perm_p": p,
                        "rate_by_alpha": rates, "random_rate": rand})
    return out
