#!/usr/bin/env python3
"""
Demo script for step-by-step formal verification.

This script demonstrates how to use the StepwisePipeline to generate
structured verification that can be formalized step by step.
"""

import json
import logging
import os
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

def run_lean_code_creation(nlv_model="qwen3", nlv_template="structured", 
                           formal_model="yinyaowenhua1314/deepseek-prover-v2-7b",
                           formal_template="in-context-lean",
                           formal_language="lean",
                           framework="ollama", 
                           separate_gpus=False,
                           split="all",
                           output_file="auto",
                           openai_token=None,
                           wandb_token=None):
    """Demonstrate step-by-step verification pipeline."""
    logger = logging.getLogger(__name__)

    try:
        # Create model providers
        logger.info("Creating model providers...")
        if openai_token:
            print("BIMBIIIIIIIIIIIIIII!!!!!!!!!!!!!!")
            os.environ["OPENAI_API_KEY"] = openai_token

        # For NLV, use Ollama (adjust model name as needed)
        if separate_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        nlv_provider = ModelFactory.create_provider(
            framework,
            nlv_model,
            api_key=openai_token
        )
        if not nlv_provider.api_key:
            print(openai_token)
            0/0

        # For formal code, use HuggingFace model
        if separate_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        if formal_model == nlv_model:
            formal_provider = nlv_provider
        else:
            formal_provider = ModelFactory.create_provider(
                framework,
                formal_model,  # Replace with a better model for code generation
                api_key=openai_token
                )

        # Create stepwise pipeline
        logger.info("Creating stepwise pipeline...")
        use_wandb=False
        if wandb_token:
            os.environ["WANDB_API_KEY"] = wandb_token
            use_wandb=True

        pipeline = FormalStepVerificationPipeline(
            nlv_provider=nlv_provider,
            formal_provider=formal_provider,
            formal_language=formal_language,
            nlv_template=nlv_template,
            formal_template=formal_template,
            use_wandb=use_wandb,
            )

        # Create a simple test dataset
        logger.info("Creating test dataset...")
        dataset_loader = DatasetFactory.create_loader("processbench", split=split)

        # Process using stepwise pipeline
        logger.info("Running in batch mode")
        results = pipeline.process_dataset(
            dataset_loader,
        )
        if output_file=="auto":
            output_file = f"processbench_results_nlv-{nlv_model}_{nlv_template}-formal-{formal_model}_{formal_template}.json"
        save_results(results, output_file)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write("Error: %s\n" % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(
            description="""Run conversion of Processbench steps into Lean code
            for verification purposes.""")
    
    parser.add_argument("--framework", "-f", default="ollama", help="the framework to use to run the experiments (only ollama supported at the moment)")
    parser.add_argument("--nlv_model", "-nlvm", default="qwen3", help="the model used for the natural language to Lean or for the informal verification step (depending on the prompt templates used)")
    parser.add_argument("--formal_model", "-formalm", default="yinyaowenhua1314/deepseek-prover-v2-7b", help="the model used for the Lean proving or for the natural language to Lean step (depending on the prompt templates used)")
    parser.add_argument("--nlv_template", "-nlvt", default="structured", help="the prompt template used for the first step of the pipeline.")
    parser.add_argument("--formal_template", "-formalt", default="in-context-lean", help="the prompt template used for the second step of the pipeline")
    parser.add_argument("--split", "-s", default="all", choices=["all", "gsm8k", "math", "olympiadbench", "omnimat"], help="the split of processbench to use. If all, use all of them.")
    parser.add_argument("--output_file", "-o", default="auto", help="name of the output file. If auto generates it by concatenating parameters used.")
    parser.add_argument("--formal_language", "-language", choices=["lean"], default="lean", help="Which formal language to use. For now, only Lean is supported.")
    parser.add_argument("--separate_gpus", action="store_true", help="Whether to load the NLV and Formal method on separate GPUs (suggested for efficiency if GPUs are big enough)")
    parser.add_argument("--openai_token", "-openai", required=False, help="The openai token required if using openai as framework.")
    parser.add_argument("--wandb_token", "-wandb", required=False, help="The wandb token required if using wandb for logging. Including this parameter enables wandb by default.")
    args = parser.parse_args()

    setup_logging()
    run_lean_code_creation(**vars(args))
