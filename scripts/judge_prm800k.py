"""Classify PRM800K rated steps with a local LLM judge: does the step assert a
mathematical claim, and is that claim true or false?  A PRM800K rating of -1
marks reasoning the labeler rejected, which is often a poor strategy rather
than a false proposition; this judge lets us keep only steps whose label
reflects the truth of a claim.  Writes external/prm800k/judged.jsonl with
problem_id, node_id, rating, text, judgement in {TRUE_CLAIM, FALSE_CLAIM,
NO_CLAIM}, raw."""
import hashlib, json, os, re, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-12b-it"
N_TRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
SRC = "/scratch/users/iacopog/leansteer/external/prm800k/phase2_test.jsonl"
OUT = "/scratch/users/iacopog/leansteer/external/prm800k/judged.jsonl"
_MATHY = re.compile(r"[=<>]|\$|\\frac|\d")
PROMPT = """You are grading one step of a written solution to a math problem.

Problem:
{problem}

Previous steps of the solution:
{prefix}

Step to grade:
{step}

Does the step to grade assert a specific mathematical claim (a computation, an equation, an identity, a numeric fact, a property of an object), and is that claim correct?
Answer with exactly one word:
TRUE_CLAIM  - the step asserts a specific mathematical claim and the claim is correct
FALSE_CLAIM - the step asserts a specific mathematical claim and the claim is incorrect (wrong calculation, wrong formula, wrong fact)
NO_CLAIM    - the step is a plan, a strategy, a restatement, a question, or otherwise does not assert a checkable mathematical claim"""

def pid(problem): return "prm-" + hashlib.sha1(problem.strip().encode()).hexdigest()[:10]

items = []
for line in open(SRC):
    r = json.loads(line); problem = r["question"]["problem"]
    chosen = []
    for i, s in enumerate(r["label"]["steps"]):
        for j, c in enumerate(s["completions"]):
            t = (c.get("text") or "").strip(); rt = c.get("rating")
            if rt in (1, -1) and not t.startswith("# Answer") and 15 <= len(t) <= 300 and _MATHY.search(t):
                items.append({"problem_id": pid(problem), "node_id": f"step_{i}_{j}", "rating": rt, "text": t,
                              "problem": problem, "prefix": "\n".join(chosen[-6:]) or "(none)"})
        idx = s.get("chosen_completion")
        if idx is not None and idx < len(s["completions"]):
            chosen.append((s["completions"][idx].get("text") or "").strip())
seen = set(); uniq = []
for it in items:
    k = (it["problem_id"], it["text"])
    if k not in seen:
        seen.add(k); uniq.append(it)
neg = [x for x in uniq if x["rating"] == -1]; pos = [x for x in uniq if x["rating"] == 1]
import random; random.seed(0); random.shuffle(pos)
items = neg + pos[:N_TRUE]
print(f"judging {len(neg)} false-rated and {min(N_TRUE, len(pos))} true-rated steps with {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL)
prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=x["problem"], prefix=x["prefix"], step=x["text"])}],
                                   tokenize=False, add_generation_prompt=True) for x in items]
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.6, seed=0)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=True)
from collections import Counter; cnt = Counter()
with open(OUT, "w") as f:
    for x, o in zip(items, outs):
        raw = o.outputs[0].text.strip()
        j = next((k for k in ("FALSE_CLAIM", "TRUE_CLAIM", "NO_CLAIM") if k in raw.upper().replace(" ", "_")), "UNPARSED")
        cnt[(x["rating"], j)] += 1
        f.write(json.dumps({k: x[k] for k in ("problem_id", "node_id", "rating", "text", "problem")} | {"judgement": j, "raw": raw}) + "\n")
print(dict(cnt), "->", OUT)
