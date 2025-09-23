"""Prompt template classes for flexible prompt generation."""

from abc import ABC, abstractmethod
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
        return self.template.format(language=language, input=input_text)


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