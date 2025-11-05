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
from formalin.core import FormalStepVerificationPipeline


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def save_results(results, output_file: str, format: str = "json"):
    """Save results to file."""
    logger = logging.getLogger(__name__)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)

        elif format.lower() == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result.to_dict(), f, ensure_ascii=False)
                    f.write('\n')

        else:
            raise ValueError(f"Unsupported output format: {format}")

        logger.info(f"Saved {len(results)} results to {output_path}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def demo_stepwise_verification():
    """Demonstrate step-by-step verification pipeline."""
    logger = logging.getLogger(__name__)

    try:
        # Create model providers
        logger.info("Creating model providers...")

        # For NLV, use Ollama (adjust model name as needed)
        nlv_provider = ModelFactory.create_provider(
            "ollama",
            "gpt-oss"
        )

        # For formal code, use HuggingFace model
        formal_provider = ModelFactory.create_provider(
            "ollama",
            "yinyaowenhua1314/deepseek-prover-v2-7b"  # Replace with a better model for code generation
        )

        # Create stepwise pipeline
        logger.info("Creating stepwise pipeline...")
        pipeline = FormalStepVerificationPipeline(
            nlv_provider=nlv_provider,
            formal_provider=formal_provider,
            formal_language="lean"
        )

        # Create a simple test dataset
        logger.info("Creating test dataset...")
        dataset_loader = DatasetFactory.create_loader("processbench")

        # Process using stepwise pipeline
        logger.info("Running in batch mode")
        results = pipeline.process_dataset(
            dataset_loader,
        )

        save_results(results, "processbench_results.json")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    setup_logging()
    demo_stepwise_verification()