"""The four FaithformBench-AF autoformalizers and their exact prompts.

System/user prompts are reproduced verbatim from the benchmark's
``veriform/formalization/fine_tuned.py`` (which in turn follows each model
card), so the residual stream we probe is the one the benchmark's verdicts
were produced from.  Chat formatting goes through each repo's own
``chat_template``; StepFun additionally gets the literal ``<think>`` appended
(the official inference code does this to switch the R1-distilled base into
reasoning mode).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    system_prompt: Optional[str]
    user_template: str          # with {step}
    assistant_prefix: str = ""  # appended after the generation prompt
    trust_remote_code: bool = False
    thinking: bool = False      # emits <think>...</think> before the Lean block
    fast_tokenizer: bool = False  # load via PreTrainedTokenizerFast (see Herald)

    def messages(self, step: str) -> List[dict]:
        msgs = []
        if self.system_prompt is not None:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": self.user_template.format(step=step)})
        return msgs


STEPFUN_HEADER = "import Mathlib\n\nopen Real\n"

MODELS: Dict[str, ModelSpec] = {
    "goedel": ModelSpec(
        key="goedel", hf_id="Goedel-LM/Goedel-Formalizer-V2-8B", system_prompt=None,
        # no separator between the statement and "Think before..." — training-time format
        user_template=(
            "Please autoformalize the following natural language problem statement in Lean 4. "
            "Use the following theorem name: my_favorite_theorem\n"
            "The natural language statement is: \n"
            "{step}"
            "Think before you provide the lean statement."),
        thinking=True),
    "kimina": ModelSpec(
        key="kimina", hf_id="AI-MO/Kimina-Autoformalizer-7B",
        system_prompt="You are an expert in mathematics and Lean 4.",
        user_template=(
            "Please autoformalize the following problem in Lean 4 with a header. "
            "Use the following theorem names: my_favorite_theorem.\n\n"
            "{step}")),
    "herald": ModelSpec(
        key="herald", hf_id="FrenzyMath/Herald_translator",
        system_prompt="You are an expert at Lean 4 and Mathematics.",
        user_template=(
            "Please translate the natural language statement to Lean4 code with the header\n"
            "**Name**\nmy_favorite_theorem\n"
            "**Informal statement**\n{step}\n"),
        # transformers >= 5 resolves this repo's `LlamaTokenizer` to the slow
        # sentencepiece path, which drops every space ("exper", "tat", ...) and
        # cannot decode; the tokenizer.json backend is the one the model was
        # trained with and matches transformers 4.x token ids exactly.
        trust_remote_code=True, fast_tokenizer=True),
    "stepfun": ModelSpec(
        key="stepfun", hf_id="stepfun-ai/StepFun-Formalizer-7B",
        system_prompt="You are an expert in mathematics and Lean 4.",
        user_template=(
            "Please autoformalize the following problem in Lean 4 with a header. "
            "Use the following theorem names: my_favorite_theorem.\n\n"
            "{step}"
            f"\n\nYour code should start with:\n```Lean4\n{STEPFUN_HEADER}\n```\n"),
        assistant_prefix="<think>", thinking=True),
}


def get_spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(MODELS)}")


def build_prompt(spec: ModelSpec, tokenizer, step: str) -> str:
    text = tokenizer.apply_chat_template(spec.messages(step), tokenize=False, add_generation_prompt=True)
    return text + spec.assistant_prefix
