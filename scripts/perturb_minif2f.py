"""Corrupt miniF2F informal statements with the benchmark's LLM-perturbation prompt
(veriform/perturbation/perturbers.py::BaseLLMPerturber.system_prompt), using a local
instruct model under vLLM.  Writes external/minif2f/perturbed.jsonl with one record per
problem: id, split, informal (true), perturbed (or null when the model returned the
original), model."""
import json, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-12b-it"
SRC = sys.argv[2] if len(sys.argv) > 2 else None      # optional statements.jsonl (id, split, informal, formal); default: miniF2F from the Hub
OUT = sys.argv[3] if len(sys.argv) > 3 else "/scratch/users/iacopog/leansteer/external/minif2f/perturbed.jsonl"
SYSTEM = ("You are a mathematical adversary. Your task is to take a specific "
          "mathematical or logical statement that is TRUE and subtly modify it "
          "so that it becomes FALSE. \n"
          "Rules:\n"
          "1. Maintain the original LaTeX formatting, variable names, and style exactly.\n"
          "2. Make the change subtle (e.g., flip an inequality, change a constant, swap a quantifier).\n"
          "3. Do NOT explain your reasoning.\n"
          "4. Output ONLY the modified statement.")
if SRC:
    rows = [json.loads(l) for l in open(SRC)]
else:
    ds = load_dataset("cat-searcher/minif2f-lean4")
    rows = [dict(id=r["id"], split=r["split"], informal=r["informal_stmt"].strip(), formal=r["formal_statement"]) for s in ds for r in ds[s]]
tok = AutoTokenizer.from_pretrained(MODEL)
# Gemma has no system role: fold the instruction into the user turn (the benchmark's Gemini path does the same)
prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SYSTEM}\n\nStatement: {r['informal']}\nFalsified Statement:"}],
                                   tokenize=False, add_generation_prompt=True) for r in rows]
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.6, seed=0)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512), use_tqdm=True)
n_changed = 0
with open(OUT, "w") as f:
    for r, o in zip(rows, outs):
        p = o.outputs[0].text.strip().strip("`").strip()
        if p.lower().startswith("falsified statement:"):
            p = p.split(":", 1)[1].strip()
        changed = bool(p) and p != r["informal"]
        n_changed += changed
        f.write(json.dumps({**r, "perturbed": p if changed else None, "model": MODEL}) + "\n")
print(f"{len(rows)} statements, {n_changed} perturbed -> {OUT}")
