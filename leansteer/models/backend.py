"""Hugging Face backend with residual-stream read/write hooks.

Adapted from REVISE's ``revise/model/backend.py``.  Differences: anchors are
arbitrary per-row *weightings over prompt positions* (so ``prompt_last``,
``step_last`` and ``step_mean`` are all captured in one pass), and generation
uses ``model.generate`` with the steering hooks left in place.

Layer indexing: ``layer`` is the residual stream *after* decoder block
``layer`` (1-based); ``0`` is the token embedding output.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from leansteer import config  # noqa: F401
from leansteer.models.registry import ModelSpec, build_prompt


@dataclass
class Intervention:
    """``steer``: add ``vector`` [d] (already scaled) at the anchor position and
    everything after it (prefill) and at every generated position.
    ``project``: set the component along unit ``vector`` to ``target``.
    ``head_steer``: like ``steer`` but on the per-head attention outputs of
    block ``layer`` (input of ``o_proj``); ``vector`` is [n_heads * head_dim] with
    zeros for heads that are left alone."""
    kind: str
    layer: int
    vector: torch.Tensor
    positions: str = "anchor_onward"   # or "all"
    target: float = 0.0


@dataclass
class ForwardOutput:
    anchors: Dict[str, np.ndarray]           # name -> [B, n_captured_layers, d] float16
    prompt_len: List[int]
    entropy: List[float]
    top_prob: List[float]
    top_token: List[str]
    timing: Dict[str, float] = field(default_factory=dict)


def load_tokenizer(spec: ModelSpec):
    """Tokenizer for ``spec`` with a round-trip check: a prompt must decode
    back to itself token-for-token, otherwise activations and generations are
    silently wrong (this happened with Herald under transformers 5)."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if spec.fast_tokenizer:
        tok = PreTrainedTokenizerFast.from_pretrained(spec.hf_id)
    else:
        tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    probe = build_prompt(spec, tok, "If a = b^k with b>0, b≠1, then log_b(a) = k.")
    ids = tok(probe, add_special_tokens=False)["input_ids"]
    if tok.decode(ids) != probe:
        raise RuntimeError(f"tokenizer for {spec.hf_id} does not round-trip the prompt: {tok.decode(ids)[:120]!r}")
    return tok


def _find_layers(model) -> torch.nn.ModuleList:
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers", "model.language_model.layers"):
        obj, ok = model, True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError("could not locate decoder layers")


class HFBackend:
    def __init__(self, spec: ModelSpec, dtype: str = "bfloat16", device: str = "cuda",
                 capture_layers: Optional[Sequence[int]] = None, attn_implementation: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.tok = load_tokenizer(spec)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        kw = dict(low_cpu_mem_usage=True, trust_remote_code=spec.trust_remote_code)
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=torch_dtype, **kw)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch_dtype, **kw)
        self.model.eval().to(device)
        self.layers = _find_layers(self.model)
        self.embed = self.model.get_input_embeddings()
        self.n_layers = len(self.layers)
        self.hidden_size = self.model.config.hidden_size
        self.capture_layers = list(capture_layers) if capture_layers is not None else list(range(self.n_layers + 1))
        gc = self.model.generation_config
        eos = gc.eos_token_id if gc is not None and gc.eos_token_id is not None else self.tok.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos)

    # ------------------------------------------------------------------ prompts
    def prompt(self, step: str) -> str:
        return build_prompt(self.spec, self.tok, step)

    def encode(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def step_span(self, prompt: str, step: str) -> Tuple[List[int], int]:
        """Token indices (into the unpadded prompt) overlapping the step text,
        and the prompt length."""
        enc = self.tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        s = prompt.rfind(step)
        if s < 0:
            return [], len(ids)
        e = s + len(step)
        idx = [i for i, (a, b) in enumerate(offs) if b > a and a < e and b > s]
        return idx, len(ids)

    # ------------------------------------------------------------------ hooks
    def _make_hook(self, layer_idx: int, anchor: int, weights: Optional[torch.Tensor],
                   store: Optional[dict], interventions: List[Intervention]):
        """``weights``: [n_anchors, B, T] float32 weighting over *padded* prompt positions."""
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            modified = False
            prefill = hs.shape[1] > 1
            for iv in interventions:
                if iv.layer != layer_idx:
                    continue
                if iv.kind == "steer":
                    if prefill and iv.positions == "generated":
                        continue
                    v = iv.vector.to(device=hs.device, dtype=hs.dtype)
                    if not modified:
                        hs = hs.clone(); modified = True
                    if prefill and iv.positions != "all":
                        hs[:, anchor:, :] = hs[:, anchor:, :] + v
                    else:
                        hs = hs + v
                elif iv.kind == "project":
                    if prefill and iv.positions == "generated":
                        continue
                    v = iv.vector.to(device=hs.device, dtype=torch.float32)
                    if not modified:
                        hs = hs.clone(); modified = True
                    sl = slice(None) if (iv.positions == "all" or not prefill) else slice(anchor, None)
                    seg = hs[:, sl, :].to(torch.float32)
                    coef = seg @ v
                    hs[:, sl, :] = (seg - (coef - iv.target)[..., None] * v).to(hs.dtype)
            if store is not None and prefill and layer_idx in store:
                T = weights.shape[-1]
                w = weights.to(device=hs.device, dtype=torch.float32)                 # [A, B, T]
                vals = torch.einsum("abt,btd->abd", w, hs[:, :T, :].to(torch.float32))
                store[layer_idx] = vals.to(torch.float16).cpu().numpy()                # [A, B, d]
            if modified:
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            return None
        return hook

    def _make_head_hook(self, layer_idx: int, weights: torch.Tensor, store: dict):
        """Pre-hook on ``self_attn.o_proj``: its input is the concatenation of the
        per-head attention outputs ``[B, T, n_heads * head_dim]`` (what ITI-style
        head probes read).  Stored as weighted sums over prompt positions."""
        def hook(module, args):
            z = args[0]
            if z.shape[1] <= 1:
                return None
            T = weights.shape[-1]
            w = weights.to(device=z.device, dtype=torch.float32)
            vals = torch.einsum("abt,btd->abd", w, z[:, :T, :].to(torch.float32))
            store[layer_idx] = vals.to(torch.float16).cpu().numpy()
            return None
        return hook

    def _make_head_steer_hook(self, anchor: int, ivs: List[Intervention]):
        """Pre-hook on ``self_attn.o_proj`` adding per-head vectors to its input."""
        def hook(module, args):
            z = args[0]
            prefill = z.shape[1] > 1
            out = z.clone()
            for iv in ivs:
                if prefill and iv.positions == "generated":
                    continue
                v = iv.vector.to(device=z.device, dtype=z.dtype)
                if prefill and iv.positions != "all":
                    out[:, anchor:, :] = out[:, anchor:, :] + v
                else:
                    out = out + v
            return (out,) + tuple(args[1:])
        return hook

    def _register(self, anchor: int, weights: Optional[torch.Tensor], store: Optional[dict],
                  interventions: List[Intervention]):
        handles = []
        resid_ivs = [iv for iv in interventions if iv.kind != "head_steer"]
        needed = set(self.capture_layers if store is not None else []) | {iv.layer for iv in resid_ivs}
        for li in needed:
            mod = self.embed if li == 0 else self.layers[li - 1]
            handles.append(mod.register_forward_hook(self._make_hook(li, anchor, weights, store, resid_ivs)))
        by_layer: Dict[int, List[Intervention]] = {}
        for iv in interventions:
            if iv.kind == "head_steer":
                by_layer.setdefault(iv.layer, []).append(iv)
        for li, ivs in by_layer.items():
            handles.append(self.layers[li - 1].self_attn.o_proj.register_forward_pre_hook(self._make_head_steer_hook(anchor, ivs)))
        return handles

    def _pad(self, enc: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B, Lp = len(enc), max(len(e) for e in enc)
        ids = torch.full((B, Lp), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((B, Lp), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, Lp - len(e):] = torch.tensor(e)
            mask[i, Lp - len(e):] = 1
        return ids, mask, Lp

    # ------------------------------------------------------------------ capture
    def head_layout(self) -> Tuple[int, int]:
        cfg = self.model.config
        n_heads = int(cfg.num_attention_heads)
        head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads)
        return n_heads, head_dim

    @torch.no_grad()
    def capture(self, prompts: Sequence[str], anchor_weights: Dict[str, List[np.ndarray]],
                interventions: Optional[Sequence[Intervention]] = None, component: str = "resid") -> ForwardOutput:
        """One prefill pass.  ``anchor_weights[name][i]`` is a weight vector over
        the *unpadded* positions of prompt ``i`` (e.g. one-hot at the last token,
        or uniform over the step tokens).  ``component``: ``resid`` (residual
        stream after each block, layer 0 = embeddings) or ``heads`` (per-head
        attention outputs of each block, layers 1..n; layer 0 is not defined)."""
        t0 = time.time()
        interventions = list(interventions or [])
        enc = [self.encode(p) for p in prompts]
        ids, mask, Lp = self._pad(enc)
        B = len(enc)
        names = list(anchor_weights)
        W = torch.zeros((len(names), B, Lp), dtype=torch.float32)
        for a, name in enumerate(names):
            for i, w in enumerate(anchor_weights[name]):
                w = np.asarray(w, dtype=np.float32)
                W[a, i, Lp - len(w):] = torch.from_numpy(w)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        anchor = Lp - 1
        layers = [li for li in self.capture_layers if component == "resid" or li >= 1]
        store = {li: None for li in layers}
        if component == "heads":
            # interventions first so that a head capture sees the steered values (pre-hooks run in registration order)
            handles = self._register(anchor, None, None, interventions)
            handles += [self.layers[li - 1].self_attn.o_proj.register_forward_pre_hook(self._make_head_hook(li, W, store)) for li in layers]
        else:
            handles = self._register(anchor, W, store, interventions)
        try:
            out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                             position_ids=pos.to(self.device), use_cache=False, logits_to_keep=1)
            logits = out.logits[:, -1, :].float()
        finally:
            for h in handles:
                h.remove()
        lp = torch.log_softmax(logits, dim=-1)
        probs = lp.exp()
        entropy = (-(probs * lp).sum(-1)).tolist()
        top_prob, top_tok = probs.max(-1)
        anchors = {}
        stacked = np.stack([store[li] for li in layers], axis=2)   # [A, B, nL, d]
        for a, name in enumerate(names):
            anchors[name] = stacked[a]
        return ForwardOutput(anchors=anchors, prompt_len=[len(e) for e in enc], entropy=entropy,
                             top_prob=top_prob.tolist(), top_token=[self.tok.decode([t]) for t in top_tok.tolist()],
                             timing={"prefill_s": time.time() - t0, "prompt_tokens": float(sum(len(e) for e in enc))})

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 1024,
                 interventions: Optional[Sequence[Intervention]] = None, **gen_kw) -> List[str]:
        """Greedy decoding (the benchmark's ``deterministic`` setting) with steering
        hooks active at the anchor (last prompt token) and every generated token."""
        interventions = list(interventions or [])
        enc = [self.encode(p) for p in prompts]
        ids, mask, Lp = self._pad(enc)
        handles = self._register(Lp - 1, None, None, interventions)
        try:
            out = self.model.generate(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos_ids,
                temperature=None, top_p=None, top_k=None, **gen_kw)
        finally:
            for h in handles:
                h.remove()
        gen = out[:, Lp:]
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen]
