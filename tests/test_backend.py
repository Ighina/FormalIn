"""Backend on a tiny random Qwen2 model with a real cached tokenizer."""
import os

import numpy as np
import pytest
import torch

from leansteer.models.backend import HFBackend, Intervention
from leansteer.models.registry import get_spec
from leansteer.run.collect import anchor_weights

os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def backend():
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    spec = get_spec("kimina")
    try:
        tok = AutoTokenizer.from_pretrained(spec.hf_id)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer not cached: {e}")
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok) + 10, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    model = Qwen2ForCausalLM(cfg).eval()
    b = HFBackend.__new__(HFBackend)
    b.spec, b.device, b.tok, b.model = spec, "cpu", tok, model
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    b.layers, b.embed, b.n_layers, b.hidden_size = model.model.layers, model.get_input_embeddings(), 3, 64
    b.capture_layers = [0, 1, 2, 3]
    b.eos_ids = [tok.eos_token_id]
    return b


def _weights(b, steps):
    prompts = [b.prompt(s) for s in steps]
    spans = [b.step_span(p, s) for p, s in zip(prompts, steps)]
    w = {a: [anchor_weights(a, n, idx) for idx, n in spans] for a in ("prompt_last", "step_last", "step_mean")}
    return prompts, spans, w


def test_step_span_and_capture_matches_unbatched(backend):
    steps = ["1 + 1 = 2.", "If a = b^k with b>0, b≠1, then log_b(a) = k (definition of logarithm)."]
    prompts, spans, w = _weights(backend, steps)
    for (idx, n), s, p in zip(spans, steps, prompts):
        assert idx and 0 < len(idx) <= n
        text = backend.tok.decode([backend.encode(p)[i] for i in idx])
        assert s.replace(" ", "") in text.replace(" ", "")
    out = backend.capture(prompts, w)
    assert out.anchors["prompt_last"].shape == (2, 4, 64)
    single = backend.capture(prompts[:1], {a: v[:1] for a, v in w.items()})
    for a in w:   # left padding must not change the captured state of the shorter prompt
        np.testing.assert_allclose(out.anchors[a][0].astype(np.float32), single.anchors[a][0].astype(np.float32), rtol=2e-2, atol=2e-2)
    # mean anchor equals the mean of per-token states
    hs = backend.model(input_ids=torch.tensor([backend.encode(prompts[0])]), output_hidden_states=True).hidden_states
    ref = hs[2][0, spans[0][0], :].mean(0).detach().numpy()
    np.testing.assert_allclose(out.anchors["step_mean"][0, 2].astype(np.float32), ref, rtol=2e-2, atol=2e-2)


def test_steering_hook_moves_residual(backend):
    steps = ["2 + 2 = 4."]
    prompts, spans, w = _weights(backend, steps)
    v = torch.randn(64)
    base = backend.capture(prompts, w).anchors["prompt_last"][0, 2].astype(np.float32)
    steered = backend.capture(prompts, w, interventions=[Intervention("steer", 2, v)]).anchors["prompt_last"][0, 2].astype(np.float32)
    np.testing.assert_allclose(steered - base, v.numpy(), rtol=5e-2, atol=5e-2)
    # layer 3 (downstream) should change too, layer 1 (upstream) must not
    up = backend.capture(prompts, w, interventions=[Intervention("steer", 2, v)]).anchors["prompt_last"][0, 1]
    np.testing.assert_array_equal(up, backend.capture(prompts, w).anchors["prompt_last"][0, 1])


def test_generate_runs(backend):
    out = backend.generate([backend.prompt("3 * 3 = 9.")], max_new_tokens=4, interventions=[Intervention("steer", 1, torch.zeros(64))])
    assert len(out) == 1 and isinstance(out[0], str)


@pytest.mark.parametrize("key", ["goedel", "kimina", "herald", "stepfun"])
def test_tokenizer_roundtrip(key):
    from leansteer.models.backend import load_tokenizer
    spec = get_spec(key)
    try:
        tok = load_tokenizer(spec)
    except OSError as e:  # pragma: no cover
        pytest.skip(f"tokenizer not cached: {e}")
    p = backend_prompt = spec.messages("256 = 2^8.")
    text = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) + spec.assistant_prefix
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == text
    assert " " in tok.decode(ids[2:10])   # spaces survive (Herald's slow tokenizer dropped them)


def test_head_capture_matches_o_proj_input(backend):
    steps = ["1 + 1 = 2.", "If a = b^k with b>0, b≠1, then log_b(a) = k."]
    prompts, spans, w = _weights(backend, steps)
    out = backend.capture(prompts, w, component="heads")
    n_heads, head_dim = backend.head_layout()
    assert out.anchors["prompt_last"].shape == (2, 3, n_heads * head_dim)   # layers 1..3 only
    captured = {}
    h = backend.layers[1].self_attn.o_proj.register_forward_pre_hook(lambda m, a: captured.setdefault("z", a[0].detach()))
    try:
        backend.model(input_ids=torch.tensor([backend.encode(prompts[0])]))
    finally:
        h.remove()
    np.testing.assert_allclose(out.anchors["prompt_last"][0, 1].astype(np.float32), captured["z"][0, -1].numpy(), rtol=2e-2, atol=2e-2)


def test_head_steer_moves_only_selected_head(backend):
    steps = ["2 + 2 = 4."]
    prompts, spans, w = _weights(backend, steps)
    n_heads, head_dim = backend.head_layout()
    v = torch.zeros(n_heads * head_dim); v[head_dim:2 * head_dim] = torch.randn(head_dim)   # head 1 of layer 2
    base = backend.capture(prompts, w, component="heads").anchors["prompt_last"][0]
    steered = backend.capture(prompts, w, component="heads", interventions=[Intervention("head_steer", 2, v)]).anchors["prompt_last"][0]
    # the hook adds to o_proj's input; the head store captures that same input, so layer 2 moves by v
    np.testing.assert_allclose((steered[1] - base[1]).astype(np.float32), v.numpy(), rtol=5e-2, atol=5e-2)
    np.testing.assert_array_equal(steered[0], base[0])   # layer 1 untouched
    # the residual stream after block 2 changes, before it does not
    r0 = backend.capture(prompts, w).anchors["prompt_last"][0]
    r1 = backend.capture(prompts, w, interventions=[Intervention("head_steer", 2, v)]).anchors["prompt_last"][0]
    assert np.abs(r1[2].astype(np.float32) - r0[2].astype(np.float32)).max() > 0 and np.array_equal(r1[1], r0[1])
