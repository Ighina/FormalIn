"""Main pipeline for formal verification."""

import logging
from typing import List, Iterator, Dict, Any, Optional

from ..models.base import BaseLLMProvider
from ..prompts.registry import PromptRegistry
from ..datasets.base import BaseDatasetLoader, DatasetItem
from .result import VerificationResult

logger = logging.getLogger(__name__)


class FormalVerificationPipeline:
    """Main pipeline for converting informal to formal verification."""

    def __init__(
        self,
        nlv_provider: BaseLLMProvider,
        formal_provider: BaseLLMProvider,
        prompt_registry: Optional[PromptRegistry] = None,
        formal_language: str = "lean",
        nlv_template: str = "default",
        formal_template: str = "default",
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
        """
        self.nlv_provider = nlv_provider
        self.formal_provider = formal_provider
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.formal_language = formal_language
        self.nlv_template_name = nlv_template
        self.formal_template_name = formal_template

        # Get templates
        self.nlv_template = self.prompt_registry.get_nlv_template(nlv_template)
        self.formal_template = self.prompt_registry.get_formal_template(formal_template)

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

        try:
            # Step 1: Generate natural language verification
            logger.debug(f"Generating NLV for problem: {item.problem[:100]}...")
            nlv_prompt = self.nlv_template.format(
                problem=item.problem,
                solution=item.solution
            )
            nlv_explanation = self.nlv_provider.generate(nlv_prompt, **nlv_params)

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
                solution=item.solution,
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
            return result

        except Exception as e:
            logger.error(f"Error processing item: {e}")
            # Return partial result with error information
            return VerificationResult(
                problem=item.problem,
                solution=item.solution,
                nlv_explanation=f"ERROR: {str(e)}",
                formal_proof="",
                metadata={**item.metadata, "error": str(e)},
                nlv_model=self.nlv_provider.model_name,
                formal_model=self.formal_provider.model_name,
                nlv_template=self.nlv_template_name,
                formal_template=self.formal_template_name,
                language=self.formal_language,
            )

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
        logger.info(f"Dataset info: {dataset_loader.get_info()}")

        results = []
        for idx, item in enumerate(dataset_loader.load()):
            logger.info(f"Processing item {idx + 1}")

            try:
                result = self.process_item(item, nlv_params, formal_params)
                results.append(result)

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
        logger.info(f"Dataset info: {dataset_loader.get_info()}")

        for idx, item in enumerate(dataset_loader.load()):
            logger.debug(f"Processing item {idx + 1}")

            try:
                result = self.process_item(item, nlv_params, formal_params)
                yield result

            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing item {idx + 1}: {e}")
                continue

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