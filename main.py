#!/usr/bin/env python3
"""
FormalIn: LLM-based informal to formal verification pipeline.

This script demonstrates the modular pipeline for converting informal mathematical
reasoning into formal verification code using LLMs.

Usage:
    python main.py [--config CONFIG_FILE] [--output OUTPUT_FILE]
"""

import argparse
import json
import logging
from pathlib import Path

from formalin.config import Config, ConfigLoader
from formalin.models import ModelFactory
from formalin.datasets import DatasetFactory
from formalin.prompts import PromptRegistry
from formalin.core import FormalVerificationPipeline


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_pipeline_from_config(config: Config) -> FormalVerificationPipeline:
    """Create pipeline from configuration."""
    logger = logging.getLogger(__name__)

    # Create model providers
    logger.info(f"Creating NLV provider: {config.nlv_model.provider}/{config.nlv_model.name}")
    nlv_provider = ModelFactory.create_provider(
        config.nlv_model.provider,
        config.nlv_model.name,
        **config.nlv_model.params
    )

    logger.info(f"Creating formal provider: {config.formal_model.provider}/{config.formal_model.name}")
    formal_provider = ModelFactory.create_provider(
        config.formal_model.provider,
        config.formal_model.name,
        **config.formal_model.params
    )

    # Create pipeline
    pipeline = FormalVerificationPipeline(
        nlv_provider=nlv_provider,
        formal_provider=formal_provider,
        formal_language=config.pipeline.formal_language,
        nlv_template=config.pipeline.nlv_template,
        formal_template=config.pipeline.formal_template,
    )

    return pipeline


def create_dataset_loader_from_config(config: Config):
    """Create dataset loader from configuration."""
    logger = logging.getLogger(__name__)

    logger.info(f"Creating dataset loader: {config.dataset.type}")
    dataset_loader = DatasetFactory.create_loader(
        config.dataset.type,
        max_items=config.dataset.max_items,
        **config.dataset.params
    )

    return dataset_loader


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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="FormalIn: Convert informal verification to formal proofs"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (overrides config)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--create-example-config",
        action="store_true",
        help="Create an example configuration file"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Create example config if requested
        if args.create_example_config:
            ConfigLoader.create_example_config("config.yaml")
            logger.info("Created example configuration file: config.yaml")
            return

        # Load configuration
        if args.config:
            config = ConfigLoader.load(args.config)
        else:
            logger.info("No config file specified, using default configuration")
            config = ConfigLoader.create_default_config()
            # Set reasonable defaults for demo
            config.dataset.max_items = 5

        # Override output file if specified
        if args.output:
            config.output_file = args.output

        # Create pipeline components
        pipeline = create_pipeline_from_config(config)
        dataset_loader = create_dataset_loader_from_config(config)

        # Run pipeline
        logger.info("Starting formal verification pipeline")

        if config.pipeline.streaming:
            logger.info("Running in streaming mode")
            results = []
            for result in pipeline.process_dataset_streaming(
                dataset_loader,
                config.nlv_generation_params,
                config.formal_generation_params
            ):
                results.append(result)
                logger.info(f"Processed item {len(results)}")

                # Print progress
                print(f"\n{'='*50}")
                print(f"Item {len(results)}")
                print(f"Problem: {result.problem[:100]}...")
                print(f"NLV: {result.nlv_explanation[:200]}...")
                print(f"Formal: {result.formal_proof[:200]}...")

        else:
            logger.info("Running in batch mode")
            results = pipeline.process_dataset(
                dataset_loader,
                config.nlv_generation_params,
                config.formal_generation_params
            )

        # Save results if output file specified
        if config.output_file:
            save_results(results, config.output_file, config.output_format)
        else:
            # Print summary
            print(f"\n{'='*50}")
            print("PIPELINE SUMMARY")
            print(f"{'='*50}")
            print(f"Processed {len(results)} items")
            print(f"NLV Model: {config.nlv_model.provider}/{config.nlv_model.name}")
            print(f"Formal Model: {config.formal_model.provider}/{config.formal_model.name}")
            print(f"Language: {config.pipeline.formal_language}")

            # Show first result as example
            if results:
                print(f"\nEXAMPLE RESULT:")
                result = results[0]
                print(f"Problem: {result.problem}")
                print(f"Solution: {result.solution}")
                print(f"NLV: {result.nlv_explanation}")
                print(f"Formal Proof: {result.formal_proof}")

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()