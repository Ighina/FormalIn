"""Main pipeline for formal verification."""

import logging
from typing import List, Iterator, Dict, Any, Optional

from ..models.base import BaseLLMProvider
from ..prompts.registry import PromptRegistry, SafeFormalTemplate
from ..datasets.base import BaseDatasetLoader, DatasetItem
from .result import VerificationResult
from dataclasses import dataclass
import re
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Logging features will be disabled.")

logger = logging.getLogger(__name__)

# @dataclass
# class VerificationStep:
#     """Represents a single verification step."""

#     step_number: int
#     title: str
#     what_to_verify: str
#     how_to_verify: str
#     required_concepts: str
#     formal_code: str = ""
#     metadata: Dict[str, Any] = None

#     def __post_init__(self):
#         if self.metadata is None:
#             self.metadata = {}


# @dataclass
# class StepwiseResult(VerificationResult):
#     """Result with step-by-step breakdown."""

#     verification_steps: List[VerificationStep] = None

#     def __post_init__(self):
#         if self.verification_steps is None:
#             self.verification_steps = []

#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary including steps."""
#         base_dict = super().to_dict()
#         base_dict["verification_steps"] = [
#             {
#                 "step_number": step.step_number,
#                 "title": step.title,
#                 "what_to_verify": step.what_to_verify,
#                 "how_to_verify": step.how_to_verify,
#                 "required_concepts": step.required_concepts,
#                 "formal_code": step.formal_code,
#                 "metadata": step.metadata,
#             }
#             for step in self.verification_steps
#         ]
#         return base_dict

@dataclass
class StepVerificationResult:
    """
    Verification Results by Individual Step
    """
    def __init__(self):
        self.steps = []

    def to_dict(self) -> Dict[str, Any]:
        base_dict = {"steps": [r.to_dict() for r in self.steps]}
        return base_dict

class FormalStepVerificationPipeline:
    """Main pipeline for converting informal to formal verification."""

    def __init__(
        self,
        nlv_provider: BaseLLMProvider,
        formal_provider: BaseLLMProvider,
        prompt_registry: Optional[PromptRegistry] = None,
        formal_language: str = "lean",
        nlv_template: str = "step",
        formal_template: str = "prover",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            nlv_provider: Provider for natural language verification
            formal_provider: Provider for formal proof generation
            prompt_registry: Registry for prompt templates
            formal_language: Target formal language (lean, coq, isabelle, etc.)
            nlv_template: Template name for NLV step
            formal_template: Template name for formal step
            use_wandb: Whether to enable wandb logging
            wandb_project: Wandb project name
            wandb_run_name: Wandb run name
            wandb_config: Additional wandb configuration
        """
        self.nlv_provider = nlv_provider
        self.formal_provider = formal_provider
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.formal_language = formal_language
        self.nlv_template_name = nlv_template
        self.formal_template_name = formal_template

        # Wandb configuration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}

        if use_wandb and not WANDB_AVAILABLE:
            logger.warning("wandb logging requested but wandb is not installed. Logging disabled.")

        # Get templates
        self.nlv_template = self.prompt_registry.get_nlv_template(nlv_template)
        self.formal_template = self.prompt_registry.get_formal_template(formal_template)

    def _init_wandb(self, dataset_info: Optional[Dict[str, Any]] = None):
        """Initialize wandb run with experiment configuration."""
        if not self.use_wandb:
            return

        # Build configuration dictionary
        config = {
            "nlv_model": self.nlv_provider.model_name,
            "formal_model": self.formal_provider.model_name,
            "formal_language": self.formal_language,
            "nlv_template": self.nlv_template_name,
            "formal_template": self.formal_template_name,
        }

        # Add dataset info if available
        if dataset_info:
            config.update({"dataset": dataset_info})

        # Add user-provided config
        config.update(self.wandb_config)

        # Initialize wandb
        wandb.init(
            project=self.wandb_project or "formal-verification",
            name=self.wandb_run_name,
            config=config,
        )
        logger.info(f"Initialized wandb run: {wandb.run.name}")

    def _log_step_results(
        self,
        step_results: StepVerificationResult,
        item_idx: int,
    ):
        """Log all step verification results for a single item to wandb."""
        if not self.use_wandb:
            return

        for step_idx, result in enumerate(step_results.steps):
            # Convert result to dict
            result_dict = result.to_dict()

            # Create log entry with step and item indices
            log_data = {
                "item_idx": item_idx,
                "step_idx": step_idx,
                f"item_{item_idx}/step_{step_idx}/nlv_explanation_length": len(result.nlv_explanation),
                f"item_{item_idx}/step_{step_idx}/formal_proof_length": len(result.formal_proof),
                f"item_{item_idx}/step_{step_idx}/has_formal_proof": bool(result.formal_proof.strip()),
            }

            # Add metadata if present
            if result.metadata:
                for key, value in result.metadata.items():
                    if isinstance(value, (int, float, bool, str)):
                        log_data[f"item_{item_idx}/step_{step_idx}/metadata/{key}"] = value

            # Log to wandb
            wandb.log(log_data)

    def process_item(
        self,
        item: DatasetItem,
        nlv_params: Optional[Dict[str, Any]] = None,
        formal_params: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Process a single dataset item.

        Args:
            item: Dataset item to process
            nlv_params: Parameters for NLV generation
            formal_params: Parameters for formal proof generation

        Returns:
            VerificationResult object
        """
        nlv_params = nlv_params or {}
        formal_params = formal_params or {}
        step_results = StepVerificationResult()
        assert isinstance(item.solution, list), "This pipeline requires a list of reasoning step as input!"
        previous_steps = ""
        for step in item.solution:
            try:
                # Step 1: Generate natural language verification
                logger.debug(f"Generating NLV for problem: {item.problem[:100]}...")
                if isinstance(self.nlv_template, SafeFormalTemplate):
                    nlv_prompt = self.nlv_template.format(
                    problem=item.problem,
                    previous_steps=previous_steps,
                    current_step=step
                    )
                else:
                    nlv_prompt = self.nlv_template.format(problem=item.problem, solution=step)
                nlv_explanation = self.nlv_provider.generate(nlv_prompt, **nlv_params)
                if re.findall("<think>", nlv_explanation):
                    nlv_explanation = nlv_explanation.split("</think>")[1]
                #if self.formal_template_name == "prover":
                #   nlv_explanation = re.findall("```lean(.+)```", nlv_explanation, flags=re.DOTALL)[-1]

                # Prompts like the default one might include the option to output False if the current
                # step can not be converted in a formal statement
                if re.findall("False", nlv_explanation):
                    formal_proof = ""
                else:
                    # Step 2: Generate formal proof
                    logger.debug(f"Generating formal proof in {self.formal_language}...")
                    formal_prompt = self.formal_template.format(
                        language=self.formal_language,
                        input_text=nlv_explanation
                    )
                    formal_proof = self.formal_provider.generate(formal_prompt, **formal_params)

                # Create result
                result = VerificationResult(
                    problem=item.problem,
                    solution=step,
                    nlv_prompt=nlv_prompt,
                    formal_prompt=formal_prompt,
                    nlv_explanation=nlv_explanation,
                    formal_proof=formal_proof,
                    metadata=item.metadata.copy(),
                    nlv_model=self.nlv_provider.model_name,
                    formal_model=self.formal_provider.model_name,
                    nlv_template=self.nlv_template_name,
                    formal_template=self.formal_template_name,
                    language=self.formal_language,
                )

                logger.debug("Successfully processed item")

            except Exception as e:
                logger.error(f"Error processing item: {e}")
                # Return partial result with error information
                result = VerificationResult(
                    problem=item.problem,
                    solution=item.solution,
                    nlv_explanation=nlv_explanation,
                    formal_proof="",
                    metadata={**item.metadata, "error": str(e)},
                    nlv_model=self.nlv_provider.model_name,
                    formal_model=self.formal_provider.model_name,
                    nlv_template=self.nlv_template_name,
                    formal_template=self.formal_template_name,
                    language=self.formal_language,
                )
            
            step_results.steps.append(result)
            previous_steps += f"\n{step}"
        return step_results

    def process_dataset(
        self,
        dataset_loader: BaseDatasetLoader,
        nlv_params: Optional[Dict[str, Any]] = None,
        formal_params: Optional[Dict[str, Any]] = None,
    ) -> List[VerificationResult]:
        """
        Process an entire dataset.

        Args:
            dataset_loader: Dataset loader to process
            nlv_params: Parameters for NLV generation
            formal_params: Parameters for formal proof generation

        Returns:
            List of VerificationResult objects
        """
        logger.info("Starting dataset processing")
        dataset_info = dataset_loader.get_info()
        logger.info(f"Dataset info: {dataset_info}")

        # Initialize wandb if enabled
        self._init_wandb(dataset_info)

        results = []
        for idx, item in enumerate(dataset_loader.load()):
            logger.info(f"Processing item {idx + 1}")

            try:
                result = self.process_item(item, nlv_params, formal_params)
                results.append(result)

                # Log to wandb if enabled
                self._log_step_results(result, idx)

                # Log progress for larger datasets
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1} items")

            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing item {idx + 1}: {e}")
                continue

        logger.info(f"Completed processing {len(results)} items")

        # Finish wandb run if enabled
        if self.use_wandb:
            wandb.finish()
            logger.info("Wandb run finished")

        return results

    def process_dataset_streaming(
        self,
        dataset_loader: BaseDatasetLoader,
        nlv_params: Optional[Dict[str, Any]] = None,
        formal_params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[VerificationResult]:
        """
        Process dataset in streaming mode (yields results one by one).

        Args:
            dataset_loader: Dataset loader to process
            nlv_params: Parameters for NLV generation
            formal_params: Parameters for formal proof generation

        Yields:
            VerificationResult objects
        """
        logger.info("Starting streaming dataset processing")
        dataset_info = dataset_loader.get_info()
        logger.info(f"Dataset info: {dataset_info}")

        # Initialize wandb if enabled
        self._init_wandb(dataset_info)

        try:
            for idx, item in enumerate(dataset_loader.load()):
                logger.debug(f"Processing item {idx + 1}")

                try:
                    result = self.process_item(item, nlv_params, formal_params)

                    # Log to wandb if enabled
                    self._log_step_results(result, idx)

                    yield result

                except KeyboardInterrupt:
                    logger.info("Processing interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error processing item {idx + 1}: {e}")
                    continue
        finally:
            # Finish wandb run if enabled
            if self.use_wandb:
                wandb.finish()
                logger.info("Wandb run finished")

    def update_config(
        self,
        formal_language: Optional[str] = None,
        nlv_template: Optional[str] = None,
        formal_template: Optional[str] = None,
    ):
        """Update pipeline configuration."""
        if formal_language:
            self.formal_language = formal_language

        if nlv_template:
            self.nlv_template_name = nlv_template
            self.nlv_template = self.prompt_registry.get_nlv_template(nlv_template)

        if formal_template:
            self.formal_template_name = formal_template
            self.formal_template = self.prompt_registry.get_formal_template(formal_template)
