"""Stage ``collect``: one prefill pass per step variant, residual stream stored
at every layer for each anchor.

Anchors (all pre-generation, so they exist for every row of the benchmark):

    prompt_last   last prompt token — the state from which the model starts
                  writing its formalization
    step_last     last token of the step text inside the prompt
    step_mean     mean over the step's tokens

Rows are processed in a fixed order (prompt length descending, for batching)
and written into pre-allocated float16 memmaps, so the stage is resumable.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from leansteer.data.faithform import load_af, unique_steps
from leansteer.experiment import Run
from leansteer.models.backend import HFBackend
from leansteer.models.registry import get_spec

log = logging.getLogger(__name__)
ANCHORS = ("prompt_last", "step_last", "step_mean")


def anchor_weights(name: str, prompt_len: int, step_idx: List[int]) -> np.ndarray:
    w = np.zeros(prompt_len, dtype=np.float32)
    if name == "prompt_last" or not step_idx:
        w[-1] = 1.0
    elif name == "step_last":
        w[step_idx[-1]] = 1.0
    elif name == "step_mean":
        w[step_idx] = 1.0 / len(step_idx)
    else:
        raise ValueError(name)
    return w


def get_backend(cfg: dict, model: str, capture_layers=None) -> HFBackend:
    return HFBackend(get_spec(model), dtype=cfg["dtype"], device=cfg["device"], capture_layers=capture_layers)


def run_collect(cfg: dict, model: str, backend: Optional[HFBackend] = None) -> Path:
    run = Run(cfg)
    steps = unique_steps(load_af(models=[model]))
    if cfg.get("limit"):
        steps = steps.sample(min(int(cfg["limit"]), len(steps)), random_state=0)
    return collect_steps(cfg, model, steps, run.acts_dir(model), backend)


def collect_steps(cfg: dict, model: str, steps, out: Path, backend: Optional[HFBackend] = None) -> Path:
    """Prompt-side anchors for an arbitrary frame of steps (uid, step_content) into ``out``."""
    cc = cfg["collect"]
    out.mkdir(parents=True, exist_ok=True)
    backend = backend or get_backend(cfg, model)
    stride = int(cc.get("layer_stride", 1))
    layers = sorted(set(range(0, backend.n_layers + 1, stride)) | {backend.n_layers})
    backend.capture_layers = layers
    anchors = list(cc["anchors"])

    prompts, spans, plens = [], [], []
    for step in steps.step_content:
        p = backend.prompt(step)
        idx, n = backend.step_span(p, step)
        prompts.append(p); spans.append(idx); plens.append(n)
    order = np.argsort(-np.asarray(plens), kind="stable")
    uids = [steps.uid.iloc[i] for i in order]
    N, nL, d = len(order), len(layers), backend.hidden_size

    index_path, prog_path, meta_path = out / "index.json", out / "progress.json", out / "meta.jsonl"
    index = {"model": model, "hf_id": backend.spec.hf_id, "uids": uids, "layers": layers, "hidden_size": d,
             "anchors": anchors, "n_layers_model": backend.n_layers}
    if index_path.exists():
        prev = json.loads(index_path.read_text())
        if prev["uids"] != uids or prev["layers"] != layers or prev["anchors"] != anchors:
            raise RuntimeError(f"{out} holds a collection with a different row order/layers/anchors; "
                               "use a new run_name or delete it")
    else:
        index_path.write_text(json.dumps(index))
    mm: Dict[str, np.memmap] = {}
    for a in anchors:
        p = out / f"{a}.npy"
        mm[a] = (np.lib.format.open_memmap(p, mode="r+") if p.exists()
                 else np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(N, nL, d)))
    n_done = json.loads(prog_path.read_text())["n_done"] if prog_path.exists() else 0
    if n_done >= N:
        log.info("[%s] collection complete (%d rows)", model, N)
        return out
    # truncate meta to the committed rows
    lines = meta_path.read_text().splitlines() if meta_path.exists() else []
    meta_f = open(meta_path, "w")
    for l in lines[:n_done]:
        meta_f.write(l + "\n")

    bs = int(cc["batch_size"])
    t0, toks = time.time(), 0
    pbar = tqdm(range(n_done, N, bs), desc=f"collect {model}", initial=n_done // bs, total=(N + bs - 1) // bs)
    for i0 in pbar:
        i1 = min(i0 + bs, N)
        rows = order[i0:i1]
        batch_prompts = [prompts[i] for i in rows]
        weights = {a: [anchor_weights(a, plens[i], spans[i]) for i in rows] for a in anchors}
        fo = backend.capture(batch_prompts, weights)
        for a in anchors:
            mm[a][i0:i1] = fo.anchors[a]
        for j, i in enumerate(rows):
            meta_f.write(json.dumps({"uid": steps.uid.iloc[i], "prompt_len": plens[i], "n_step_tokens": len(spans[i]),
                                     "entropy": fo.entropy[j], "top_prob": fo.top_prob[j], "top_token": fo.top_token[j]}) + "\n")
        meta_f.flush()
        for a in anchors:
            mm[a].flush()
        prog_path.write_text(json.dumps({"n_done": i1, "N": N}))
        toks += fo.timing["prompt_tokens"]
        pbar.set_postfix(tok_s=f"{toks / max(1e-6, time.time() - t0):.0f}")
    meta_f.close()
    log.info("[%s] collected %d rows x %d layers x %d anchors in %.0fs", model, N, nL, len(anchors), time.time() - t0)
    return out


def run_collect_tf(cfg: dict, model: str, backend: Optional[HFBackend] = None) -> Path:
    """Teacher-forced anchors on the *unperturbed* rows, whose raw autoformalizer
    response the benchmark file keeps: the prompt is followed by the model's own
    recorded output and the residual stream is read at

        output_last      last token of the recorded output
        statement_last   last token of the extracted ``theorem ... sorry`` block
                         (falls back to output_last; ``has_statement`` in meta)
        statement_mean   mean over the statement's tokens (same fallback)

    Outputs longer than ``tf.max_output_tokens`` are skipped (``skipped`` in index)."""
    from leansteer.data.faithform import load_af
    run = Run(cfg)
    tc = cfg["tf"]
    out = run.model_dir(model) / "acts_tf"; out.mkdir(exist_ok=True)
    df = load_af(models=[model], with_outputs=True)
    df = df[(~df.is_perturbed) & df.autoformalizer_output.notna()].drop_duplicates("uid").reset_index(drop=True)
    if cfg.get("limit"):
        df = df.sample(min(int(cfg["limit"]), len(df)), random_state=0).reset_index(drop=True)
    backend = backend or get_backend(cfg, model)
    stride = int(tc.get("layer_stride", 1))
    layers = sorted(set(range(0, backend.n_layers + 1, stride)) | {backend.n_layers})
    backend.capture_layers = layers
    anchors = list(tc["anchors"])
    max_tokens = int(tc["max_output_tokens"])

    texts, spans, lens, keep, has_stmt = [], [], [], [], []
    for step, output, stmt in zip(df.step_content, df.autoformalizer_output, df.statement):
        p = backend.prompt(step)
        o = output
        if backend.spec.assistant_prefix and o.startswith(backend.spec.assistant_prefix):
            o = o[len(backend.spec.assistant_prefix):]
        full = p + o
        n_prompt = len(backend.encode(p))
        n_full = len(backend.encode(full))
        if n_full - n_prompt > max_tokens or n_full - n_prompt < 1:
            keep.append(False); texts.append(full); spans.append([]); lens.append(n_full); has_stmt.append(False)
            continue
        idx: List[int] = []
        if isinstance(stmt, str) and stmt in o:
            idx, _ = backend.step_span(full, stmt)
        keep.append(True); texts.append(full); spans.append(idx); lens.append(n_full); has_stmt.append(bool(idx))
    keep = np.asarray(keep)
    order = np.flatnonzero(keep)[np.argsort(-np.asarray(lens)[keep], kind="stable")]
    uids = [df.uid.iloc[i] for i in order]
    N, nL, d = len(order), len(layers), backend.hidden_size
    log.info("[%s] teacher-forced collection: %d rows (%d skipped as too long), %d with statement span", model, N,
             int((~keep).sum()), int(sum(has_stmt[i] for i in order)))

    index_path, prog_path, meta_path = out / "index.json", out / "progress.json", out / "meta.jsonl"
    index = {"model": model, "hf_id": backend.spec.hf_id, "uids": uids, "layers": layers, "hidden_size": d, "anchors": anchors,
             "n_layers_model": backend.n_layers, "skipped": [df.uid.iloc[i] for i in np.flatnonzero(~keep)]}
    if index_path.exists():
        prev = json.loads(index_path.read_text())
        if prev["uids"] != uids or prev["layers"] != layers or prev["anchors"] != anchors:
            raise RuntimeError(f"{out} holds a different collection; use a new run_name or delete it")
    else:
        index_path.write_text(json.dumps(index))
    mm: Dict[str, np.memmap] = {}
    for a in anchors:
        p = out / f"{a}.npy"
        mm[a] = (np.lib.format.open_memmap(p, mode="r+") if p.exists()
                 else np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(N, nL, d)))
    n_done = json.loads(prog_path.read_text())["n_done"] if prog_path.exists() else 0
    if n_done >= N:
        return out
    lines = meta_path.read_text().splitlines() if meta_path.exists() else []
    meta_f = open(meta_path, "w")
    for l in lines[:n_done]:
        meta_f.write(l + "\n")

    def weights(name, n, idx):
        w = np.zeros(n, dtype=np.float32)
        if name == "output_last" or not idx:
            w[-1] = 1.0
        elif name == "statement_last":
            w[idx[-1]] = 1.0
        elif name == "statement_mean":
            w[idx] = 1.0 / len(idx)
        else:
            raise ValueError(name)
        return w

    bs = int(tc["batch_size"])
    t0, toks = time.time(), 0
    pbar = tqdm(range(n_done, N, bs), desc=f"collect-tf {model}", initial=n_done // bs, total=(N + bs - 1) // bs)
    for i0 in pbar:
        i1 = min(i0 + bs, N)
        rows = order[i0:i1]
        # token budget: long outputs get smaller effective batches
        fo = backend.capture([texts[i] for i in rows], {a: [weights(a, lens[i], spans[i]) for i in rows] for a in anchors})
        for a in anchors:
            mm[a][i0:i1] = fo.anchors[a]
        for j, i in enumerate(rows):
            meta_f.write(json.dumps({"uid": df.uid.iloc[i], "n_tokens": lens[i], "has_statement": has_stmt[i],
                                     "n_statement_tokens": len(spans[i])}) + "\n")
        meta_f.flush()
        for a in anchors:
            mm[a].flush()
        prog_path.write_text(json.dumps({"n_done": i1, "N": N}))
        toks += fo.timing["prompt_tokens"]
        pbar.set_postfix(tok_s=f"{toks / max(1e-6, time.time() - t0):.0f}")
    meta_f.close()
    return out


def run_collect_heads(cfg: dict, model: str, backend: Optional[HFBackend] = None) -> Path:
    """Attention-head store: per-head outputs (input of ``o_proj``) at the prompt-side
    anchors, every layer 1..n.  Shape ``[N, n_layers, n_heads * head_dim]``; head ``h``
    of layer ``l`` is the slice ``[h*head_dim:(h+1)*head_dim]`` of row ``l-1``."""
    run = Run(cfg)
    hc = cfg["heads"]
    out = run.model_dir(model) / "acts_heads"; out.mkdir(exist_ok=True)
    steps = unique_steps(load_af(models=[model]))
    if cfg.get("limit"):
        steps = steps.sample(min(int(cfg["limit"]), len(steps)), random_state=0)
    backend = backend or get_backend(cfg, model)
    stride = int(hc.get("layer_stride", 1))
    layers = sorted(set(range(1, backend.n_layers + 1, stride)) | {backend.n_layers})
    backend.capture_layers = layers
    anchors = list(hc["anchors"])
    n_heads, head_dim = backend.head_layout()

    prompts, spans, plens = [], [], []
    for step in steps.step_content:
        p = backend.prompt(step)
        idx, n = backend.step_span(p, step)
        prompts.append(p); spans.append(idx); plens.append(n)
    order = np.argsort(-np.asarray(plens), kind="stable")
    uids = [steps.uid.iloc[i] for i in order]
    N, nL, d = len(order), len(layers), n_heads * head_dim

    index_path, prog_path = out / "index.json", out / "progress.json"
    index = {"model": model, "hf_id": backend.spec.hf_id, "uids": uids, "layers": layers, "hidden_size": d, "anchors": anchors,
             "n_layers_model": backend.n_layers, "n_heads": n_heads, "head_dim": head_dim, "component": "heads"}
    if index_path.exists():
        prev = json.loads(index_path.read_text())
        if prev["uids"] != uids or prev["layers"] != layers or prev["anchors"] != anchors:
            raise RuntimeError(f"{out} holds a different collection; use a new run_name or delete it")
    else:
        index_path.write_text(json.dumps(index))
    mm: Dict[str, np.memmap] = {}
    for a in anchors:
        p = out / f"{a}.npy"
        mm[a] = (np.lib.format.open_memmap(p, mode="r+") if p.exists()
                 else np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(N, nL, d)))
    n_done = json.loads(prog_path.read_text())["n_done"] if prog_path.exists() else 0
    if n_done >= N:
        return out
    bs = int(hc["batch_size"])
    t0 = time.time()
    for i0 in tqdm(range(n_done, N, bs), desc=f"collect-heads {model}", initial=n_done // bs, total=(N + bs - 1) // bs):
        i1 = min(i0 + bs, N)
        rows = order[i0:i1]
        fo = backend.capture([prompts[i] for i in rows], {a: [anchor_weights(a, plens[i], spans[i]) for i in rows] for a in anchors},
                             component="heads")
        for a in anchors:
            mm[a][i0:i1] = fo.anchors[a]
            mm[a].flush()
        prog_path.write_text(json.dumps({"n_done": i1, "N": N}))
    log.info("[%s] head store: %d rows x %d layers x %d heads x %d in %.0fs", model, N, nL, n_heads, head_dim, time.time() - t0)
    return out


def load_acts(run: Run, model: str, anchor: str, store: str = "acts"):
    """memmap [N, nL, d], uid->row, index dict."""
    out = run.model_dir(model) / store
    index = json.loads((out / "index.json").read_text())
    prog = json.loads((out / "progress.json").read_text())
    if prog["n_done"] < prog["N"]:
        raise RuntimeError(f"collection for {model} incomplete ({prog['n_done']}/{prog['N']})")
    X = np.load(out / f"{anchor}.npy", mmap_mode="r")
    return X, {u: i for i, u in enumerate(index["uids"])}, index
