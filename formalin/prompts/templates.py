"""Prompt template classes for flexible prompt generation."""

from abc import ABC, abstractmethod
import re
from typing import Dict, Any


class PromptTemplate(ABC):
    """Base class for prompt templates."""

    def __init__(self, template: str):
        self.template = template

    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the template with given parameters."""
        pass


class NLVTemplate(PromptTemplate):
    """Template for natural language verification prompts."""

    def format(self, problem: str, solution: str) -> str:
        """Format NLV template with problem and solution."""
        return self.template.format(problem=problem, solution=solution)


class FormalTemplate(PromptTemplate):
    """Template for formal verification prompts."""

    def format(self, language: str, input_text: str) -> str:
        """Format formal template with language and input."""
        try:
            return self.template.format(language=language, input=input_text)
        except KeyError:
            return self.template.format(input_text=input_text)

class SafeFormalTemplate(PromptTemplate):
    """Template for formalization in Safe-paper style"""

    def format(self, problem:str, previous_steps:str, current_step:str) -> str:
        templ = re.sub("<problem>", problem, self.template)
        templ = re.sub("<previous_steps>", previous_steps, templ)
        return re.sub("<current_step>", current_step, templ)
    
class ProverTemplate(PromptTemplate):
    """Template for formalization in Safe-paper style"""

    def format(self, input_text: str, **kwargs) -> str:
        lean_code = re.findall("```lean(.+)```", input_text, flags=re.DOTALL)[-1]
        #lean_code=input_text.strip()
        return self.template.format(input_text=lean_code)

# Default templates
DEFAULT_NLV_TEMPLATE = """Explain in plain and exact language how to formally verify that the solution to the problem provided below is indeed correct.
# Problem
{problem}
# Solution
{solution}"""

DEFAULT_FORMAL_TEMPLATE = """Translate the given requirements into {language}'s syntax and semantics. You only need to return the {language} formal specification without explanation.
# Input
{input}"""

# Alternative templates
DETAILED_NLV_TEMPLATE = """Given the mathematical problem and its solution below, provide a detailed step-by-step explanation of how to formally verify the correctness of this solution. Include what mathematical properties, theorems, or logical rules would need to be checked.

Problem:
{problem}

Solution:
{solution}

Please structure your verification approach as:
1. Key mathematical concepts involved
2. Logical steps to verify
3. Potential edge cases or assumptions to check"""

CONCISE_FORMAL_TEMPLATE = """Convert to {language} syntax:
{input}"""

VERBOSE_FORMAL_TEMPLATE = """You are an expert in formal verification using {language}. Convert the following informal mathematical verification into a complete formal proof in {language}, including all necessary definitions, lemmas, and proof steps.

Informal verification:
{input}

Please provide the complete {language} code with:
1. All necessary imports and dependencies
2. Relevant definitions and lemmas
3. The main theorem statement
4. Complete proof"""

# Formalin step-wise template
OLD_STEP_NLV_TEMPLATE = """Analyze the mathematical problem and a single step from the proposed solution below, then provide a verification plan for the given step.

# Problem
{problem}

# Solution Step
{solution}

Please provide your verification analysis in the following format:

**What to verify:** [Specific claim or property to check]
**How to verify:** [Detailed explanation of verification approach]
**Required concepts:** [Mathematical concepts, theorems, or definitions needed]

BE FAITHFUL TO THE INPUT SOLUTION, EVEN IF YOU THINK IT IS INCORRECT!!!
YOUR TASK IS TO DEVISE A PLAN TO VERIFY THE SOLUTION STEP, NOT TO CORRECT IT!!!
ONLY GENERATE THE PLAN FOR THE GIVEN SOLUTION STEP, DO NOT ATTEMPT TO GENERATE PLANS FOR ANY OTHER FUTURE OR PAST STEP!!!
IF THERE IS NOTHING TO VERIFY (E.G. "THE RESULT IS 36") THEN OUTPUT A VERIFICATION PLAN STATING TO JUST RETURN TRUE IN THE FORMALIZATION PROCESS!

The output should:
- Clearly specify what needs to be proven
- Include all necessary mathematical details
- Be suitable for individual formalization in a proof assistant
- Be faithful to the input

Verification Plan:"""

STEP_NLV_TEMPLATE = """You are a Step-by-Step Logical Validator. Your goal is to convert a single step of natural language reasoning into a formal verification plan.

### STRICT CONSTRAINTS
1. SCOPE: You must ONLY analyze the text provided under "TARGET STEP". Ignore the final goal of the main problem except for context.
2. NO LOOKAHEAD: Do not generate plans for future steps. Do not solve the full problem. If the step says "X is 5", do not calculate what Y is.
3. FAITHFULNESS: Verify exactly what is written. If the step makes a claim, your plan must verify that specific claim, even if it seems trivial.
4. TRIVIALITY: If the step is a simple declaration (e.g., "Let x = 5"), the plan should be to "Define variable x and assign value 5".

### INPUT DATA

[CONTEXT / FULL PROBLEM DESCRIPTION]
{problem}

[TARGET STEP TO VERIFY]
{solution}

### OUTPUT FORMAT
Provide the verification plan for the TARGET STEP only.

**Verification Goal:** [Concise statement of the logical transition in this step]
**Formalization Strategy:** [How to translate this text into a proof state]
**What to verify:** [Specific claim or equality to check]
**How to verify:** [Detailed explanation: e.g., "Check if variable P_fri is defined as 18"]
**Required concepts:** [List concepts]

### YOUR RESPONSE"""

# Step-by-step templates
STRUCTURED_NLV_TEMPLATE = """Analyze the mathematical problem and its solution below, then provide a structured verification plan broken down into clear, independent steps.

# Problem
{problem}

# Solution
{solution}

Please provide your verification analysis in the following format:

## STEP 1: [Brief step title]
**What to verify:** [Specific claim or property to check]
**How to verify:** [Detailed explanation of verification approach]
**Required concepts:** [Mathematical concepts, theorems, or definitions needed]

## STEP 2: [Brief step title]
**What to verify:** [Specific claim or property to check]
**How to verify:** [Detailed explanation of verification approach]
**Required concepts:** [Mathematical concepts, theorems, or definitions needed]

[Continue with additional steps as needed...]

## FINAL STEP: Conclusion
**What to verify:** [Final claim that all steps combine to prove]
**How to verify:** [How the individual steps combine to complete the verification]
**Required concepts:** [Any final logical principles needed]

BE FAITHFUL TO THE INPUT SOLUTION, EVEN IF YOU THINK IT IS INCORRECT!!!
YOUR TASK IS TO DEVISE A PLAN TO VERIFY THE SOLUTION, NOT TO CORRECT IT!!!

Each step should be:
- Independent and self-contained
- Clearly specify what needs to be proven
- Include all necessary mathematical details
- Be suitable for individual formalization"""

STEP_FORMAL_TEMPLATE = """You are an expert in formal verification using {language}. Convert the following single verification step into {language} code.

# Verification Step
{input}

Requirements:
1. Focus only on formalizing this specific step
2. Include necessary definitions and lemmas for this step only
3. Provide the formal statement and proof
4. Use clear, readable {language} syntax
5. Add comments explaining the formalization

Return only the {language} code without additional explanation."""

STEP_LEAN_TEMPLATE = """You are a Lean 4 expert. This is a basic tutorial on how to use Lean 4: /- This file is intended for Lean beginners. The goal is to demonstrate what it feels like to prove things using Lean and mathlib. Complicated definitions and theory building are not covered. Everything is covered again more slowly and with exercises in the next files. -/ -- We want real numbers and their basic properties import Mathlib.Data.Real.Basic -- We want to be able to use Lean's built-in "help" functionality import Mathlib.Tactic.LibrarySearch -- We want to be able to define functions using the law of excluded middle noncomputable section /- Our first goal is to define the set of upper bounds of a set of real numbers. This is already defined in mathlib (in a more general context), but we repeat it for the sake of exposition. Right-click "upperBounds" below to get offered to jump to mathlib's version -/ #check upperBounds /-- The set of upper bounds of a set of real numbers ℝ -/ def upBounds (A : Set ℝ) := { x : ℝ | ∀ a ∈ A, a ≤ x } /-- Predicate is_maximum a A means a is a maximum of A -/ def IsMaximum (a : ℝ) (A : Set ℝ) := a ∈ A ∧ a ∈ upBounds A /- In the above definition, the symbol ∧ means "and". We also see the most visible difference between set theoretic foundations and type theoretic ones (used by almost all proof assistants). In set theory, everything is a set, and the only relation you get from foundations are = and ∈. In type theory, there is a meta-theoretic relation of "typing": a : ℝ reads "a is a real number" or, more precisely, "the type of a is ℝ". Here "meta-theoretic" means this is not a statement you can prove or disprove inside the theory, it's a fact that is true or not. Here we impose this fact, in other circumstances, it would be checked by the Lean kernel. By contrast, a ∈ A is a statement inside the theory. Here it's part of the definition, in other circumstances it could be something proven inside Lean. -/ /- For illustrative purposes, we now define an infix version of the above predicate. It will allow us to write a is_a_max_of A, which is closer to a sentence. -/ infixl:55 " is_a_max_of " => IsMaximum /- Let's prove something now! A set of real numbers has at most one maximum. Here everything left of the final : is introducing the objects and assumption. The equality x = y right of the colon is the conclusion. -/ theorem unique_max (A : Set ℝ) (x y : ℝ) (hx : x is_a_max_of A) (hy : y is_a_max_of A) : x = y := by -- We first break our assumptions in their two constituent pieces. -- We are free to choose the name following with rcases hx with ⟨x_in, x_up⟩ rcases hy with ⟨y_in, y_up⟩ -- Assumption x_up means x isn't less than elements of A, let's apply this to y specialize x_up y -- Assumption x_up now needs the information that y is indeed in A. specialize x_up y_in -- Let's do this quicker with roles swapped specialize y_up x x_in -- We explained to Lean the idea of this proof. -- Now we know x ≤ y and y ≤ x, and Lean shouldn't need more help. -- linarith proves equalities and inequalities that follow linearly from -- the assumption we have. linarith /- The above proof is too long, even if you remove comments. We don't really need the unpacking steps at the beginning; we can access both parts of the assumption hx : x is_a_max_of A using shortcuts hx.1 and hx.2. We can also improve readability without assistance from the tactic state display, clearly announcing intermediate goals using have. This way we get to the following version of the same proof. -/ example (A : Set ℝ) (x y : ℝ) (hx : x is_a_max_of A) (hy : y is_a_max_of A) : x = y := by have : x ≤ y := hy.2 x hx.1 have : y ≤ x := hx.2 y hy.1 linarith /- Notice how mathematics based on type theory treats the assumption ∀ a ∈ A, a ≤ y as a function turning an element a of A into the statement a ≤ y. More precisely, this assumption is the abbreviation of ∀ a : ℝ, a ∈ A → a ≤ y. The expression hy.2 x appearing in the above proof is then the statement x ∈ A → x ≤ y, which itself is a function turning a statement x ∈ A into x ≤ y so that the full expression hy.2 x hx.1 is indeed a proof of x ≤ y. One could argue a three-line-long proof of this lemma is still two lines too long. This is debatable, but mathlib's style is to write very short proofs for trivial lemmas. Those proofs are not easy to read but they are meant to indicate that the proof is probably not worth reading. In order to reach this stage, we need to know what linarith did for us. It invoked the lemma le_antisymm which says: x ≤ y → y ≤ x → x = y. This arrow, which is used both for function and implication, is right associative. So the statement is x ≤ y → (y ≤ x → x = y) which reads: I will send a proof p of x ≤ y to a function sending a proof q' of y ≤ x to a proof of x = y. Hence le_antisymm p q' is a proof of x = y. Using this we can get our one-line proof: -/ example (A : Set ℝ) (x y : ℝ) (hx : x is_a_max_of A) (hy : y is_a_max_of A) : x = y := le_antisymm (hy.2 x hx.1) (hx.2 y hy.1) /- Such a proof is called a proof term (or a "term mode" proof). Notice it has no by. It is directly the kind of low level proof that the Lean kernel is consuming. Commands like rcases, specialize or linarith are called tactics, they help users constructing proof terms that could be very tedious to write directly. The most efficient proof style combines tactics with proof terms like our previous have : x ≤ y := hy.2 x hx.1 where hy.2 x hx.1 is a proof term embeded inside a tactic mode proof. In the remaining of this file, we'll be characterizing infima of sets of real numbers in term of sequences. -/ /-- The set of lower bounds of a set of real numbers ℝ -/ def lowBounds (A : Set ℝ) := { x : ℝ | ∀ a ∈ A, x ≤ a } /- We now define a is an infimum of A. Again there is already a more general version in mathlib. -/ def IsInf (x : ℝ) (A : Set ℝ) := x is_a_max_of lowBounds A -- Let's define it also as an infix operator infixl:55 " is_an_inf_of " => IsInf /- We need to prove that any number which is greater than the infimum of A is greater than some element of A. -/ theorem inf_lt {A : Set ℝ} {x : ℝ} (hx : x is_an_inf_of A) : ∀ y, x < y → ∃ a ∈ A, a < y := by -- Let y be any real number. intro y -- Let's prove the contrapositive contrapose -- The symbol ¬ means negation. Let's ask Lean to rewrite the goal without negation, -- pushing negation through quantifiers and inequalities push_neg -- Let's assume the premise, calling the assumption h intro h -- h is exactly saying y is a lower bound of A so the second part of -- the infimum assumption hx applied to y and h is exactly what we want. exact hx.2 y h /- In the above proof, the sequence contrapose, push_neg is so common that it can be abbreviated to contrapose!. With these commands, we enter the gray zone between proof checking and proof finding. Practical computer proof checking crucially needs the computer to handle tedious proof steps. In the next proof, we'll start using linarith a bit more seriously, going one step further into automation. Our next real goal is to prove inequalities for limits of sequences. We extract the following lemma: if y ≤ x + ε for all positive ε then y ≤ x. -/ theorem le_of_le_add_eps {x y : ℝ} : (∀ ε > 0, y ≤ x + ε) → y ≤ x := by -- Let's prove the contrapositive, asking Lean to push negations right away. contrapose! -- Assume h : x < y. intro h -- We need to find ε such that ε is positive and x + ε < y. -- Let's use (y-x)/2 use (y - x) / 2 -- we now have two properties to prove. Let's do both in turn, using linarith constructor linarith linarith /- Note how linarith was used for both sub-goals at the end of the above proof. We could have shortened that using the semi-colon combinator instead of comma, writing constructor <;> linarith. Next we will study a compressed version of that proof: -/ example {x y : ℝ} : (∀ ε > 0, y ≤ x + ε) → y ≤ x := by contrapose! exact fun h => ⟨(y - x) / 2, by linarith, by linarith⟩ /- The angle brackets ⟨ and ⟩ introduce compound data or proofs. A proof of a ∃ z, P z statemement is composed of a witness z₀ and a proof h of P z₀. The compound is denoted by ⟨z₀, h⟩. In the example above, the predicate is itself compound, it is a conjunction P z ∧ Q z. So the proof term should read ⟨z₀, ⟨h₁, h₂⟩⟩ where h₁ (resp. h₂) is a proof of P z₀ (resp. Q z₀). But these so-called "anonymous constructor" brackets are right-associative, so we can get rid of the nested brackets. Note also how we can use by to enter tactics anywhere a term is expected. Going all the way to a proof term would make the proof much longer, because we crucially use automation with contrapose! and linarith. -/ /- One could argue that the above proof is a bit too terse, and we are relying too much on linarith. Let's have more linarith calls for smaller steps. For the sake of (tiny) variation, we will also assume the premise and argue by contradiction instead of contraposing. -/ example {x y : ℝ} : (∀ ε > 0, y ≤ x + ε) → y ≤ x := by intro h -- Assume the conclusion is false, and call this assumption H. by_contra H push_neg at H -- Now let's compute. have key := calc -- Each line must end with := followed by a proof term -- We want to specialize our assumption h to ε = (y-x)/2 but this is long to -- type, so let's put a hole _ that Lean will fill in by comparing the -- statement we want to prove and our proof term with a hole. As usual, -- positivity of (y-x)/2 is proved by linarith y ≤ x + (y - x) / 2 := h _ (by linarith) _ = x / 2 + y / 2 := by ring _ < y := by linarith -- our key now says y < y (notice how the sequence ≤, =, < was correctly -- merged into a <). Let linarith find the desired contradiction now. linarith -- alternatively, we could have provided the proof term -- exact lt_irrefl y key /- Now we are ready for some analysis. Let's define convergence of sequences of real numbers (of course there is a much more general definition in mathlib). -/ /-- The sequence u tends to l -/ def Limit (u : ℕ → ℝ) (l : ℝ) := ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - l| ≤ ε /- In the above definition, u n denotes the n-th term of the sequence. We can add parentheses to get u (n) but we try to avoid parentheses because they pile up very quickly (and note the space between u and ( is required). -/ -- If y ≤ u n for all n and u n goes to x then y ≤ x theorem le_lim {x y : ℝ} {u : ℕ → ℝ} (hu : Limit u x) (ineq : ∀ n, y ≤ u n) : y ≤ x := by -- Let's apply our previous lemma apply le_of_le_add_eps -- We need to prove y ≤ x + ε for all positive ε. -- Let ε be any positive real intro ε ε_pos -- we now specialize our limit assumption to this ε, and immediately -- fix a N as promised by the definition. rcases hu ε ε_pos with ⟨N, HN⟩ -- Now we only need to compute until reaching the conclusion calc y ≤ u N := ineq N _ = x + (u N - x) := by linarith -- In the next step we use the gcongr tactic which uses "generalized congruence" lemmas -- to zoom on the relevant part of the inequality goal, in this case u N - x ≤ |u N - x|. -- We then need a lemma saying z ≤ |z|. Because we don't know the name of this lemma, -- let's use exact?. Because searching through the library can be slow, -- Lean will write what it found in the Lean message window when cursor is on -- that line, so that we can replace it by the lemma. We see le_abs_self, which -- says a ≤ |a|, exactly what we're looking for. _ ≤ x + |u N - x| := by gcongr ; exact? _ ≤ x + ε := by gcongr ; apply HN; linarith /- The next lemma has been extracted from the main proof in order to discuss numbers. In ordinary maths, we know that ℕ is *not* contained in ℝ, whatever the construction of real numbers that we use. For instance a natural number is not an equivalence class of Cauchy sequences. But it's very easy to pretend otherwise. Formal maths requires slightly more care. In the statement below, the "type ascription" (n + 1 : ℝ) forces Lean to convert the natural number n+1 into a real number. The "inclusion" map will be displayed in tactic state as ↑. There are various lemmas asserting this map is compatible with addition and monotone, but we don't want to bother writing their names. The norm_cast tactic is designed to wisely apply those lemmas for us. -/ theorem inv_succ_pos : ∀ n : ℕ, 1 / (n + 1 : ℝ) > 0 := by -- Let n be any integer intro n -- Since we don't know the name of the relevant lemma, asserting that the inverse of -- a positive number is positive, let's state that is suffices -- to prove that n+1, seen as a real number, is positive, and ask exact? suffices (n + 1 : ℝ) > 0 by exact? -- Now we want to reduce to a statement about natural numbers, not real numbers -- coming from natural numbers. norm_cast -- and then get the usual help from linarith linarith /- That was a pretty long proof for an obvious fact. And stating it as a lemma feels stupid, so let's find a way to write it on one line in case we want to include it in some other proof without stating a lemma. First the exact? call above displays the name of the relevant lemma: one_div_pos. We can also replace the linarith call on the last line by exact? to learn the name of the lemma Nat.succ_pos asserting that the successor of a natural number is positive. There is also a variant on norm_cast that combines it with exact. The term mode analogue of intro is fun. We get down to: -/ example : ∀ n : ℕ, 1 / (n + 1 : ℝ) > 0 := fun n ↦ one_div_pos.mpr (by exact_mod_cast Nat.succ_pos n) /- The next proof uses mostly known things, so we will commment only new aspects. -/ theorem limit_inv_succ : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 1 / (n + 1 : ℝ) ≤ ε := by intro ε ε_pos suffices ∃ N : ℕ, 1 / ε ≤ N by -- Because we didn't provide a name for the above statement, Lean called it this. -- Let's fix an N that works. rcases this with ⟨N, HN⟩ use N intro n Hn -- Now we want to rewrite the goal using lemmas -- div_le_iff' : 0 < b → (a / b ≤ c ↔ a ≤ b * c) -- div_le_iff : 0 < b → (a / b ≤ c ↔ a ≤ c * b) -- the second one will be rewritten from right to left, as indicated by ←. -- Lean will create a side goal for the required positivity assumption that -- we don't provide for div_le_iff'. rw [div_le_iff', ← div_le_iff ε_pos] -- We want to replace assumption Hn by its real counter-part so that -- linarith can find what it needs. replace Hn : (N : ℝ) ≤ n exact_mod_cast Hn linarith -- we are still left with the positivity assumption. We already discussed -- how to prove it in the preceding lemma, but we could alternatively use -- the positivity tactic whose job is to prove obvious positivity statements. positivity -- Now we need to prove that sufficient statement. -- We want to use that ℝ is archimedean. So we start typing -- exact archimedean_ and hit Ctrl-space to see what completion Lean proposes -- the lemma archimedean_iff_nat_le sounds promising. We select the left to -- right implication using .1. This a generic lemma for fields equiped with -- a linear (ie total) order. We need to provide a proof that ℝ is indeed -- archimedean. This is done using the infer_instance tactic that will be -- covered elsewhere. exact archimedean_iff_nat_le.1 (by infer_instance) (1 / ε) /- We can now put all pieces together, with almost no new things to explain. -/ theorem inf_seq (A : Set ℝ) (x : ℝ) : x is_an_inf_of A ↔ x ∈ lowBounds A ∧ ∃ u : ℕ → ℝ, Limit u x ∧ ∀ n, u n ∈ A := by constructor · intro h constructor · exact h.1 -- On the next line, we don't need to tell Lean to treat n+1 as a real number because -- we add x to it, so Lean knows there is only one way to make sense of this expression. have key : ∀ n : ℕ, ∃ a ∈ A, a < x + 1 / (n + 1) := by intro n -- we can use the lemma we proved above apply inf_lt h -- and another one we proved! have : 0 < 1 / (n + 1 : ℝ) := inv_succ_pos n linarith -- Now we need to use axiom of (countable) choice choose u hu using key use u constructor · intro ε ε_pos -- again we use a lemma we proved, specializing it to our fixed ε, and fixing a N rcases limit_inv_succ ε ε_pos with ⟨N, H⟩ use N intro n hn have : x ≤ u n := h.1 _ (hu n).1 have := calc u n < x + 1 / (n + 1) := (hu n).2 _ ≤ x + ε := add_le_add (le_refl x) (H n hn) rw [abs_of_nonneg] <;> linarith · intro n exact (hu n).1 · intro h -- Assumption h is made of nested compound statements. We can use -- rcases to unpack it in one go. rcases h with ⟨x_min, u, lim, huA⟩ constructor exact x_min intro y y_mino apply le_lim lim intro n exact y_mino (u n) (huA n) 
Based on this, follow the following verification instructions expressed in natural language to create a Lean 4 proof: note that the lean proof can also be incorrect. DO NOT CORRECT ANY CLAIM. DO NOT OUPUT ANYTHING BUT LEAN 4 CODE: {input_text}
"""

OLD_IN_CONTEXT_LEAN_TEMPLATE = """You are a translator that converts natural language mathematical verification procedures into executable Lean 4 code. Your goal is to produce code that **actually verifies** whether mathematical claims are true or false by computation or proof.

## Core Requirements

1. **ALWAYS verify the claim computationally or by proof** - never assume claims as hypotheses
2. **Use `rfl` (reflexivity) or `decide` for computational verification** when possible
3. **ALWAYS use `import Mathlib` as the only import** - do not use specific submodule imports
4. **Output ONLY the Lean 4 code** - no explanations, predictions, or additional content
5. **DO NOT use `sorry`, axioms, or unproven assumptions** - all proofs must be complete
6. **DO NOT change equations or values** from the natural language input - translate exactly as given
7. **The code must succeed for TRUE claims and FAIL for FALSE claims** - this is how we verify correctness

## Lean 4 Syntax Notes

- **CRITICAL**: This is Lean 4, NOT Lean 3
- Type constants are UpperCamelCase: `Nat`, `List`, `Int`, `Real`
- Lambda expressions use `=>`: `fun x => x`
- Many Mathlib names are UpperCamelCase: `Fintype`, `Finset`
- **DO NOT use `in` keyword from Lean 3** - Lean 4 uses different syntax for let bindings
- Use `let x := value; rest` NOT `let x := value in rest`

## Translation Strategy

For **computational claims** (arithmetic, concrete values):
- Use `example : claim := rfl` or tactics to verify
- Lean will compute both sides and check equality

For **mathematical properties** requiring proof:
- State as `theorem` or `example`
- Prove using Mathlib tactics: `ring`, `norm_num`, `simp`, `omega`, etc.

---

## Examples

### Example 1: True Arithmetic Claim

**Natural Language Input:**
```
Claim: 103 raised to the 6th power equals 1224238819633
How To Verify: Compute 103^6 and check if it equals 1224238819633
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

def Root6 (x y : Nat) : Prop := y ^ 6 = x

-- This will SUCCEED because 103^6 actually equals 1224238819633
example : Root6 1224238819633 103 := by
  unfold Root6
  norm_num
```

---

### Example 2: False Arithmetic Claim

**Natural Language Input:**
```
Claim: 103 raised to the 6th power equals 1061520150601
How To Verify: Compute 103^6 and check if it equals 1061520150601
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

def Root6 (x y : Nat) : Prop := y ^ 6 = x

-- This will FAIL because 103^6 does not equal 1061520150601
example : Root6 1061520150601 103 := by
  unfold Root6
  norm_num
```

**What happens:** Lean 4 will throw an error because `norm_num` cannot prove a false statement.

---

### Example 3: True Simple Equality

**Natural Language Input:**
```
Claim: 15 + 27 equals 42
How To Verify: Add 15 and 27 and compare to 42
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

-- This will SUCCEED
example : (15 : Nat) + 27 = 42 := rfl
```

---

### Example 4: False Simple Equality

**Natural Language Input:**
```
Claim: 15 + 27 equals 40
How To Verify: Add 15 and 27 and compare to 40
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

-- This will FAIL with a type mismatch error
example : (15 : Nat) + 27 = 40 := rfl
```

---

### Example 5: True Divisibility Property

**Natural Language Input:**
```
Claim: 24 is divisible by 6
How To Verify: Check if there exists a natural number k such that 6 * k = 24
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

-- This will SUCCEED
example : ∃ k : Nat, 6 * k = 24 := by
  use 4
  norm_num
```

---

### Example 6: False Divisibility Property

**Natural Language Input:**
```
Claim: 25 is divisible by 6
How To Verify: Check if there exists a natural number k such that 6 * k = 25
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

-- This will FAIL - no such k exists
example : ∃ k : Nat, 6 * k = 25 := by
  use 4  -- or any number
  norm_num  -- This will fail
```

---

### Example 7: True Inequality

**Natural Language Input:**
```
Claim: 100 is less than 200
How To Verify: Compare 100 and 200
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

-- This will SUCCEED
example : (100 : Nat) < 200 := by decide
```

---

### Example 8: False Inequality

**Natural Language Input:**
```
Claim: 200 is less than 100
How To Verify: Compare 200 and 100
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

-- This will FAIL
example : (200 : Nat) < 100 := by decide
```

---

### Example 9: Let Binding (Lean 4 Syntax)

**Natural Language Input:**
```
Claim: Given x = 5 and y = 10, x + y equals 15
How To Verify: Define x and y, then compute x + y
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

-- This will SUCCEED
-- Note: Lean 4 uses semicolon, NOT "in" keyword
example : let x := 5; let y := 10; x + y = 15 := by
  norm_num
```

**WRONG (Lean 3 syntax):**
```lean
-- DO NOT USE THIS - this is Lean 3 syntax
example : let x := 5 in let y := 10 in x + y = 15 := by
  norm_num
```

---

### Example 10: Multiple Step Verification

**Natural Language Input:**
```
Claim: The square of 12 equals 144
How To Verify: Compute 12 * 12 and verify it equals 144
```

**Correct Lean 4 Translation:**
```lean
import Mathlib

def Square (x : Nat) : Nat := x * x

-- This will SUCCEED
example : Square 12 = 144 := by
  unfold Square
  norm_num
```

---

## Common Tactics for Verification

- `rfl` - for definitional equality (computations)
- `norm_num` - for numerical/arithmetic goals
- `decide` - for decidable propositions (inequalities, divisibility, etc.)
- `ring` - for polynomial ring equations
- `omega` - for linear arithmetic over integers
- `simp` - for simplification using lemmas

## Key Principle

**NEVER write code like this (WRONG):**
```lean
theorem claim (h : false_statement) : false_statement := h
```

**ALWAYS write code like this (CORRECT):**
```lean
example : statement := by
  -- actual proof here using tactics
  norm_num  -- or rfl, decide, etc.
```

## Import Guidelines

**ALWAYS use:**
```lean
import Mathlib
```

**NEVER use specific imports like:**
```lean
import Mathlib.Data.Nat.Basic  -- DON'T USE
import Mathlib.Tactic.Ring     -- DON'T USE
```

The proof must **actually verify** the claim through computation or logical derivation, not assume it as a hypothesis.
INPUT: {input_text}"""

IN_CONTEXT_LEAN_TEMPLATE = """You are a translator that converts natural language mathematical verification plans into executable Lean 4 code. Your goal is to produce code that **actually verifies** whether mathematical claims are true or false by computation or proof.

## Core Requirements

1. **ALWAYS verify the claim computationally or by proof** - never assume claims as hypotheses.
2. **Use `rfl` (reflexivity) or `decide` for computational verification** when possible.
3. **ALWAYS use `import Mathlib` as the only import** - do not use specific submodule imports.
4. **Output ONLY the Lean 4 code** - no explanations, predictions, or additional content.
5. **DO NOT use `sorry`, axioms, or unproven assumptions** - all proofs must be complete.
6. **DO NOT change equations or values** - translate exactly as given.
7. **The code must succeed for TRUE claims and FAIL for FALSE claims** - this is how we verify correctness.

## Handling Incomplete Input (Token Limits)

Sometimes the input text provided to you may be truncated (cut off) due to token limits.
- **DO NOT** attempt to complete the English sentence.
- **DO NOT** repeat the English text.
- **DO** attempt to generate the valid Lean 4 code based on the mathematical intent visible so far.
- If the "What to verify" section is cut off, look at the "Verification Goal" or context to infer the intended check.

## Output Format

You must begin your response with the delimiter:
`### START LEAN CODE ###`
Followed immediately by the Lean code.

## Code Style Guide (STRICT)
1. **NO LINE WRAPPING:** Do not wrap long lines. Write code on a single line even if it is long.
2. **ATOMIC IDENTIFIERS:** Never insert a newline inside a variable name (e.g., write `totalFirstPhase`, NOT `total\nFirstPhase`).
3. **ONE STATEMENT PER LINE:** Each definition or theorem must be on its own line.

---

## Examples

### Example 1: True Arithmetic Claim

*INPUT PLAN:*

**Verification Goal:** Verify the power calculation.
**Formalization Strategy:** Define the exponentiation relation and check equality.
**What to verify:** 103 raised to the 6th power equals 1224238819633
**How to verify:** Compute 103^6 and check if it matches the target value.
**Required concepts:** Exponentiation, Nat.

### START LEAN CODE ###
```Lean

import Mathlib

def Root6 (x y : Nat) : Prop := y ^ 6 = x

-- This will SUCCEED because 103^6 actually equals 1224238819633
example : Root6 1224238819633 103 := by
  unfold Root6
  norm_num```

### Example 2: False Arithmetic Claim
*INPUT PLAN:*

**Verification Goal:** Verify the power calculation.
**Formalization Strategy:** Define the exponentiation.
**What to verify:** 103 raised to the 6th power equals 1061520150601
**How to verify:** Compute 103^6 and check for equality.

### START LEAN CODE ###
```Lean

import Mathlib

def Root6 (x y : Nat) : Prop := y ^ 6 = x

-- This will FAIL because 103^6 does not equal 1061520150601
example : Root6 1061520150601 103 := by
  unfold Root6
  norm_num```
### Example 3: Simple Equality (Truncated Input)
*INPUT PLAN:*

**Verification Goal:** Check simple addition.
**What to verify:** 15 + 27 equals 42
**How to verify:** Add 15 and 27 and com
(Note: Input cut off mid-sentence)

### START LEAN CODE ###
```Lean

import Mathlib

-- Input was truncated, but intent (equality check) is clear
example : (15 : Nat) + 27 = 42 := rfl```
### Example 4: Divisibility Property
*INPUT PLAN:*

**Verification Goal:** Verify divisibility.
**Formalization Strategy:** Use existential quantifier.
**What to verify:** 24 is divisible by 6
**How to verify:** Check if there exists a natural number k such that 6 * k = 24

### START LEAN CODE ###
```Lean

import Mathlib

-- This will SUCCEED
example : ∃ k : Nat, 6 * k = 24 := by
  use 4
  norm_num```
### Example 5: Let Binding (Lean 4 Syntax)
*INPUT PLAN:*

**Verification Goal:** Variable assignment and summation.
**What to verify:** Given x = 5 and y = 10, verify x + y equals 15
**How to verify:** Define x and y using let bindings, then sum them.

### START LEAN CODE ###
```Lean

import Mathlib

-- Note: Lean 4 uses semicolon, NOT "in" keyword
example : let x := 5; let y := 10; x + y = 15 := by
  norm_num```
### Example 6: Multi-step Calculation (Heavily Truncated)
*INPUT PLAN:*

**Verification Goal:** Verify calculation step.
**What to verify:** 100 - 20 = 80
**How to verify:** Perform subtra

### START LEAN CODE ###
```Lean

import Mathlib

-- Input truncated. Formalizing the visible equality claim.
example : 100 - 20 = 80 := rfl```
## Lean 4 Syntax Reminders
Use let x := v; (semicolon) NOT let x := v in

Use norm_num for arithmetic.

Use decide for inequalities.

Use fun x => for lambdas.

*INPUT PLAN:* 

{input_text}

### START LEAN CODE ###
"""

OLD_STEPS_SAFE_TEMPLATE = """Given a question and the steps to answer it, you need to determine whether the final step of the answer may involve a hallucination that requires theorem proving in Lean 4.
* If the step is simple and intuitive, and you are confident that it does not need verification, please answer False.
* However, you need to verify ** ALL NUMERICAL ** operations, no matter how simple or intuitive they may seem.
* If the step has a certain leap that is not very intuitive and may involve a hallucination, please provide a Lean theorem that can verify the step.
* This Lean 4 theorem should support the step; if the Lean 4 theorem can be proven, then the step is correct and does not involve a hallucination.
* Ensure that the Lean theorems you provide ** CONFORM ** to the syntax of Lean 4, and ** AVOID USING NATURAL LANGUAGE ** to describe properties.
* Do ** NOT ** provide a proof method for the theorem; you can use "sorry" as a placeholder.
* Output the formalized theorem of the final step or False, and do ** NOT ** output any other content or predict next step.
* Note that each step is derived from the previous ones, so the theorem may require referencing information from the question or earlier steps.

Note that Lean 4 is not backward compatible with Lean 3.
* Type constants are now in UpperCamelCase, for example, `Nat` and `List`. Many variables in Mathlib have also changed to UpperCamelCase, such as `fintype` becoming `Fintype`.
* Lambda expressions now use `=>` as the separator. For example, `fun x => x` is the identity function, instead of `λ x, x`.

### Question:
Let \[f(x) = \left\{
\begin\{array\}\{cl\} ax+3, &\text\{ if \}x>2, \\
x-5 &\text\{ if \} -2 \le x \le 2, \\
2x-b &\text\{ if \} x <-2.
\end\{array\}
\right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).

### Step to be verified:
For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$.
### Lean:
False

### Step to be verified:
For example, $ax+3$ and $x-5$ must be equal when $x=2$.
This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$.
### Lean:
```lean
theorem test
  (a x: ℝ)
  (h₀: a * x + 3 = x - 5)
  (h₁: x = 3):
  (a = (-3)) := by sorry
```

### Step to be verified:
Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. 
Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$.
### Lean:
```lean
theorem test
  (b x: ℝ)
  (h₀: x - 5 = 2 * x - b)
  (h₁: x = -2):
  (b = 3) := by sorry
```

### Step to be verified:
So $a+b=-3+3=\boxed{0}$.
### Lean:
```lean
theorem test
  (a b: ℝ)
  (h₀: a = (-3))
  (h₁: b = 3):
  (a + b = 0) := by sorry
```

### Question:
Find the remainder when the sum \[75+76+77+78+79+80+81+82\]is divided by 16.

### Step to be verified:
We notice that 16 divides $78+82$ as well as $79+81$ and also 80.
### Lean:
```lean
theorem test:
  (16 ∣ 78 + 82) ∧ (16 ∣ 79 + 81) := by sorry
```

### Step to be verified:
Therefore the sum is congruent to  \[75+76+77\pmod{16}.\]
### Lean:
```lean
theorem test:
  (75 + 76 + 77 + 78 + 79 + 80 + 81 + 82) ≡ (75 + 76 + 77) [MOD 16] := by sorry
```

### Step to be verified:
Since these numbers are congruent to $-5$, $-4$, and $-3$ modulo 16, this can be computed as  \[-5-4-3\equiv-12\pmod{16}.\]
### Lean:
```lean
theorem test:
  (75 ≡ -5 [ZMOD 16]) ∧ (76 ≡ -4 [ZMOD 16]) ∧ (77 ≡ -3 [ZMOD 16]) ∧ (-5-4-3 = -12) := by sorry
```

### Step to be verified:
Finally, since $-12\equiv4\pmod{16}$ the remainder we seek is $\boxed{4}$.
### Lean:
```lean
theorem test:
  (-12 ≡ 4 [ZMOD 16]) ∧ (4 < 16) := by sorry
```

### Question
<problem>

### Steps that do not require verification:
<previous_steps>
### Step to be verified:
<current_step>
### Lean:"""

STEPS_SAFE_TEMPLATE = """Given a question and the steps to answer it, you need to provide a Lean theorem that can verify the step.
* This Lean 4 theorem should support the step; if the Lean 4 theorem can be proven, then the step is correct and does not involve a hallucination.
* Ensure that the Lean theorems you provide ** CONFORM ** to the syntax of Lean 4, and ** AVOID USING NATURAL LANGUAGE ** to describe properties.
* Do ** NOT ** provide a proof method for the theorem; you can use "sorry" as a placeholder.
* Output the formalized theorem of the final step, and do ** NOT ** output any other content or predict next step.
* Note that each step is derived from the previous ones, so the theorem may require referencing information from the question or earlier steps.

Note that Lean 4 is not backward compatible with Lean 3.
* Type constants are now in UpperCamelCase, for example, `Nat` and `List`. Many variables in Mathlib have also changed to UpperCamelCase, such as `fintype` becoming `Fintype`.
* Lambda expressions now use `=>` as the separator. For example, `fun x => x` is the identity function, instead of `λ x, x`.

### Question:
Let \[f(x) = \left\{
\begin{array}{cl} ax+3, &\text{ if }x>2, \\
x-5 &\text{ if } -2 \le x \le 2, \\
2x-b &\text{ if } x <-2.
\end{array}
\right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).

### Step to be verified:
For example, $ax+3$ and $x-5$ must be equal when $x=2$.
This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$.
### Lean:
```lean
theorem test
  (a x: ℝ)
  (h₀: a * x + 3 = x - 5)
  (h₁: x = 3):
  (a = (-3)) := by sorry
```

### Step to be verified:
Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. 
Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$.
### Lean:
```lean
theorem test
  (b x: ℝ)
  (h₀: x - 5 = 2 * x - b)
  (h₁: x = -2):
  (b = 3) := by sorry
```

### Step to be verified:
So $a+b=-3+3=\boxed{0}$.
### Lean:
```lean
theorem test
  (a b: ℝ)
  (h₀: a = (-3))
  (h₁: b = 3):
  (a + b = 0) := by sorry
```

### Question:
Find the remainder when the sum \[75+76+77+78+79+80+81+82\]is divided by 16.

### Step to be verified:
We notice that 16 divides $78+82$ as well as $79+81$ and also 80.
### Lean:
```lean
theorem test:
  (16 ∣ 78 + 82) ∧ (16 ∣ 79 + 81) := by sorry
```

### Step to be verified:
Therefore the sum is congruent to  \[75+76+77\pmod{16}.\]
### Lean:
```lean
theorem test:
  (75 + 76 + 77 + 78 + 79 + 80 + 81 + 82) ≡ (75 + 76 + 77) [MOD 16] := by sorry
```

### Step to be verified:
Since these numbers are congruent to $-5$, $-4$, and $-3$ modulo 16, this can be computed as  \[-5-4-3\equiv-12\pmod{16}.\]
### Lean:
```lean
theorem test:
  (75 ≡ -5 [ZMOD 16]) ∧ (76 ≡ -4 [ZMOD 16]) ∧ (77 ≡ -3 [ZMOD 16]) ∧ (-5-4-3 = -12) := by sorry
```

### Step to be verified:
Finally, since $-12\equiv4\pmod{16}$ the remainder we seek is $\boxed{4}$.
### Lean:
```lean
theorem test:
  (-12 ≡ 4 [ZMOD 16]) ∧ (4 < 16) := by sorry
```

### Question
<problem>

### Steps that do not require verification:
<previous_steps>
### Step to be verified:
<current_step>
### Lean:"""

PROVER_TEMPLATE = """Complete the following Lean 4 code:

```lean4
{input_text}
```

```lean4"""
