#!/usr/bin/env python3
"""
Demo script for step-by-step formal verification.

This script demonstrates how to use the StepwisePipeline to generate
structured verification that can be formalized step by step.
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path to import formalin
sys.path.append(str(Path(__file__).parent.parent))

from formalin.models import ModelFactory
from formalin.datasets import DatasetFactory
from formalin.core import StepwisePipeline


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demo_stepwise_verification():
    """Demonstrate step-by-step verification pipeline."""
    logger = logging.getLogger(__name__)

    try:
        # Create model providers
        logger.info("Creating model providers...")

        # For NLV, use Ollama (adjust model name as needed)
        nlv_provider = ModelFactory.create_provider(
            "ollama",
            "llama3.1"
        )

        # For formal code, use HuggingFace model
        formal_provider = ModelFactory.create_provider(
            "huggingface",
            "microsoft/DialoGPT-medium"  # Replace with a better model for code generation
        )

        # Create stepwise pipeline
        logger.info("Creating stepwise pipeline...")
        pipeline = StepwisePipeline(
            nlv_provider=nlv_provider,
            formal_provider=formal_provider,
            formal_language="lean"
        )

        # Create a simple test dataset
        logger.info("Creating test dataset...")
        test_data = [
            {
                "problem": "Prove that the sum of the first n natural numbers is n(n+1)/2",
                "solution": "We can prove this by mathematical induction. "
                           "Base case: n=1, sum = 1 = 1(1+1)/2 = 1. "
                           "Inductive step: assume true for n=k, then for n=k+1: "
                           "sum = k(k+1)/2 + (k+1) = (k+1)(k+2)/2."
            }
        ]

        # Process using stepwise pipeline
        for idx, data in enumerate(test_data):
            logger.info(f"Processing problem {idx + 1}")

            # Create dataset item
            from formalin.datasets.base import DatasetItem
            item = DatasetItem(
                problem=data["problem"],
                solution=data["solution"],
                metadata={"index": idx}
            )

            # Process with stepwise pipeline
            result = pipeline.process_item(
                item,
                formalize_steps=True  # Set to False to skip formal code generation
            )

            # Display results
            print(f"\n{'='*60}")
            print(f"PROBLEM {idx + 1}")
            print(f"{'='*60}")
            print(f"Problem: {result.problem}")
            print(f"Solution: {result.solution}")
            print(f"\nStructured Verification:")
            print(f"{'='*40}")
            print(result.nlv_explanation)

            print(f"\nVerification Steps:")
            print(f"{'='*40}")
            for step in result.verification_steps:
                print(f"\nSTEP {step.step_number}: {step.title}")
                print(f"What to verify: {step.what_to_verify}")
                print(f"How to verify: {step.how_to_verify[:200]}...")
                print(f"Required concepts: {step.required_concepts}")
                if step.formal_code:
                    print(f"Formal code:\n{step.formal_code[:300]}...")

            # Get step summary
            summary = pipeline.get_step_summary(result)
            print(f"\nSUMMARY:")
            print(f"Total steps: {summary['total_steps']}")
            print(f"Step titles: {summary['step_titles']}")
            print(f"Concepts used: {summary['concepts_used']}")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    setup_logging()
    demo_stepwise_verification()