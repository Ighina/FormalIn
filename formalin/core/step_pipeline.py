"""Step-by-step pipeline for granular formal verification."""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..models.base import BaseLLMProvider
from ..prompts.registry import PromptRegistry
from ..datasets.base import DatasetItem
from .result import VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class VerificationStep:
    """Represents a single verification step."""

    step_number: int
    title: str
    what_to_verify: str
    how_to_verify: str
    required_concepts: str
    formal_code: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StepwiseResult(VerificationResult):
    """Result with step-by-step breakdown."""

    verification_steps: List[VerificationStep] = None

    def __post_init__(self):
        if self.verification_steps is None:
            self.verification_steps = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including steps."""
        base_dict = super().to_dict()
        base_dict["verification_steps"] = [
            {
                "step_number": step.step_number,
                "title": step.title,
                "what_to_verify": step.what_to_verify,
                "how_to_verify": step.how_to_verify,
                "required_concepts": step.required_concepts,
                "formal_code": step.formal_code,
                "metadata": step.metadata,
            }
            for step in self.verification_steps
        ]
        return base_dict


class StepwisePipeline:
    """Pipeline for step-by-step formal verification."""

    def __init__(
        self,
        nlv_provider: BaseLLMProvider,
        formal_provider: BaseLLMProvider,
        prompt_registry: Optional[PromptRegistry] = None,
        formal_language: str = "lean",
        nlv_template: str = "structured",
        formal_template: str = "step",
    ):
        """
        Initialize the stepwise pipeline.

        Args:
            nlv_provider: Provider for natural language verification
            formal_provider: Provider for formal proof generation
            prompt_registry: Registry for prompt templates
            formal_language: Target formal language
        """
        self.nlv_provider = nlv_provider
        self.formal_provider = formal_provider
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.formal_language = formal_language

        # Get templates
        try:
            self.nlv_template = self.prompt_registry.get_nlv_template(nlv_template)
        except ValueError:
            logger.warning(
                f"NLV template '{nlv_template}' not found. Using default 'structured'."
            )
            self.nlv_template = self.prompt_registry.get_nlv_template("structured")
        try:
            self.formal_template = self.prompt_registry.get_formal_template(
                formal_template
            )
        except ValueError:
            logger.warning(
                f"Formal template '{formal_template}' not found. Using default 'step'."
            )
            self.formal_template = self.prompt_registry.get_formal_template("step")

    def parse_structured_verification(self, nlv_text: str) -> List[VerificationStep]:
        """Parse structured NLV output into verification steps."""
        steps = []

        # Pattern to match step sections
        step_pattern = r"## STEP (\d+|FINAL STEP): (.+?)\n\*\*What to verify:\*\* (.+?)\n\*\*How to verify:\*\* (.+?)\n\*\*Required concepts:\*\* (.+?)(?=\n##|\n\n|\Z)"

        matches = re.findall(step_pattern, nlv_text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            step_num_str, title, what_to_verify, how_to_verify, required_concepts = (
                match
            )

            # Handle "FINAL STEP" case
            if step_num_str == "FINAL STEP":
                step_number = len(steps) + 1
                title = "Conclusion"
            else:
                step_number = int(step_num_str)

            step = VerificationStep(
                step_number=step_number,
                title=title.strip(),
                what_to_verify=what_to_verify.strip(),
                how_to_verify=how_to_verify.strip(),
                required_concepts=required_concepts.strip(),
            )
            steps.append(step)

        if not steps:
            logger.warning("Could not parse structured verification into steps")
            # Fallback: create a single step with the entire text
            steps.append(
                VerificationStep(
                    step_number=1,
                    title="Complete Verification",
                    what_to_verify="Complete verification of the solution",
                    how_to_verify=nlv_text,
                    required_concepts="Various mathematical concepts",
                )
            )

        logger.info(f"Parsed {len(steps)} verification steps")
        return steps

    def formalize_step(self, step: VerificationStep, **formal_params) -> str:
        """Convert a single verification step to formal code."""
        # Create step input for formalization
        step_input = f"""
Title: {step.title}
What to verify: {step.what_to_verify}
How to verify: {step.how_to_verify}
Required concepts: {step.required_concepts}
"""

        formal_prompt = self.formal_template.format(
            language=self.formal_language, input_text=step_input.strip()
        )

        try:
            formal_code = self.formal_provider.generate(formal_prompt, **formal_params)
            return formal_code
        except Exception as e:
            logger.error(f"Error formalizing step {step.step_number}: {e}")
            return f"-- Error formalizing step: {str(e)}"

    def process_item(
        self,
        item: DatasetItem,
        nlv_params: Optional[Dict[str, Any]] = None,
        formal_params: Optional[Dict[str, Any]] = None,
        formalize_steps: bool = True,
    ) -> StepwiseResult:
        """
        Process a single dataset item with step-by-step verification.

        Args:
            item: Dataset item to process
            nlv_params: Parameters for NLV generation
            formal_params: Parameters for formal proof generation
            formalize_steps: Whether to generate formal code for each step

        Returns:
            StepwiseResult object
        """
        nlv_params = nlv_params or {}
        formal_params = formal_params or {}

        try:
            # Step 1: Generate structured NLV
            logger.debug(
                f"Generating structured NLV for problem: {item.problem[:100]}..."
            )
            nlv_prompt = self.nlv_template.format(
                problem=item.problem, solution=item.solution
            )
            nlv_text = self.nlv_provider.generate(nlv_prompt, **nlv_params)

            # Step 2: Parse into verification steps
            logger.debug("Parsing verification steps...")
            verification_steps = self.parse_structured_verification(nlv_text)

            # Step 3: Formalize each step (optional)
            if formalize_steps:
                logger.debug(f"Formalizing {len(verification_steps)} steps...")
                for step in verification_steps:
                    logger.debug(f"Formalizing step {step.step_number}: {step.title}")
                    step.formal_code = self.formalize_step(step, **formal_params)

            # Combine all formal code
            formal_proof = "\n\n".join(
                [
                    f"-- Step {step.step_number}: {step.title}\n{step.formal_code}"
                    for step in verification_steps
                    if step.formal_code
                ]
            )

            # Create result
            result = StepwiseResult(
                problem=item.problem,
                solution=item.solution,
                nlv_explanation=nlv_text,
                formal_proof=formal_proof,
                verification_steps=verification_steps,
                metadata=item.metadata.copy(),
                nlv_model=self.nlv_provider.model_name,
                formal_model=self.formal_provider.model_name,
                nlv_template="structured",
                formal_template="step",
                language=self.formal_language,
            )

            logger.debug("Successfully processed item with step-by-step verification")
            return result

        except Exception as e:
            logger.error(f"Error processing item: {e}")
            # Return partial result with error information
            return StepwiseResult(
                problem=item.problem,
                solution=item.solution,
                nlv_explanation=f"ERROR: {str(e)}",
                formal_proof="",
                verification_steps=[],
                metadata={**item.metadata, "error": str(e)},
                nlv_model=self.nlv_provider.model_name,
                formal_model=self.formal_provider.model_name,
                nlv_template="structured",
                formal_template="step",
                language=self.formal_language,
            )

    def get_step_summary(self, result: StepwiseResult) -> Dict[str, Any]:
        """Get a summary of the verification steps."""
        return {
            "total_steps": len(result.verification_steps),
            "step_titles": [step.title for step in result.verification_steps],
            "formalized_steps": len(
                [s for s in result.verification_steps if s.formal_code]
            ),
            "concepts_used": list(
                set(
                    [
                        concept.strip()
                        for step in result.verification_steps
                        for concept in step.required_concepts.split(",")
                        if concept.strip()
                    ]
                )
            ),
        }
