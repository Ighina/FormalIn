# FormalIn

LLM-based pipeline for converting informal mathematical verification into formal proof code.

## Overview

FormalIn is a modular system that uses Large Language Models to:

1. **Natural Language Verification (NLV)**: Generate detailed explanations of how to formally verify mathematical solutions
2. **Formal Proof Generation**: Convert the informal verification into formal proof code (Lean, Coq, Isabelle, etc.)

## Features

- **Flexible Model Support**: Works with both Ollama (local) and HuggingFace models
- **Multiple Datasets**: Built-in support for FormalStep dataset, plus custom dataset loading
- **Configurable Prompts**: Multiple prompt templates for different verification styles
- **Step-by-Step Verification**: Structured prompts that break verification into independent steps
- **Individual Step Formalization**: Convert each verification step to formal code separately
- **Modular Architecture**: Easy to extend and customize
- **Configuration System**: YAML/JSON configuration files for reproducible experiments

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default settings (processes 5 items from FormalStep)
python main.py

# Create an example configuration file
python main.py --create-example-config

# Run with custom configuration
python main.py --config config.yaml --output results.json
```

### Configuration

Create a configuration file to customize the pipeline:

```yaml
nlv_model:
  provider: "ollama"
  name: "llama3.1"

formal_model:
  provider: "huggingface"
  name: "fm-universe/deepseek-coder-7b-instruct-v1.5-fma"

dataset:
  type: "formalstep"
  max_items: 10

pipeline:
  formal_language: "lean"
  nlv_template: "detailed"
  formal_template: "verbose"

output_file: "results.json"
```

## Architecture

### Modules

- **`formalin.prompts`**: Flexible prompt template system
- **`formalin.models`**: LLM provider abstractions (Ollama, HuggingFace)
- **`formalin.datasets`**: Dataset loading and processing
- **`formalin.core`**: Main pipeline logic
- **`formalin.config`**: Configuration management

### Pipeline Flow

1. Load dataset (FormalStep or custom)
2. For each problem-solution pair:
   - Generate natural language verification explanation
   - Convert explanation to formal proof code
3. Save results in specified format

## Custom Datasets

You can use custom datasets in JSON, JSONL, or CSV format:

```python
from formalin.datasets import DatasetFactory

# Load custom dataset
loader = DatasetFactory.create_from_file(
    "my_dataset.json",
    problem_field="problem",
    solution_field="solution"
)
```

## Custom Models

Add support for new model providers:

```python
from formalin.models import BaseLLMProvider, ModelFactory

class MyProvider(BaseLLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        return generated_text

ModelFactory.register_provider("myprovider", MyProvider)
```

## Step-by-Step Verification

FormalIn includes a specialized `StepwisePipeline` for granular verification:

### Using the Structured Template

The `"structured"` NLV template generates verification broken into clear steps:

```yaml
pipeline:
  nlv_template: "structured"
  formal_template: "step"
```

This produces output like:

```
## STEP 1: Base Case Verification
**What to verify:** The formula holds for n=1
**How to verify:** Calculate sum for n=1 and check against formula
**Required concepts:** Basic arithmetic, substitution

## STEP 2: Inductive Hypothesis
**What to verify:** Assume formula holds for n=k
**How to verify:** State the inductive assumption clearly
**Required concepts:** Mathematical induction, logical assumptions
```

### Stepwise Pipeline Usage

```python
from formalin.core import StepwisePipeline

pipeline = StepwisePipeline(
    nlv_provider=nlv_model,
    formal_provider=formal_model,
    formal_language="lean"
)

# Process with individual step formalization
result = pipeline.process_item(item, formalize_steps=True)

# Access individual steps
for step in result.verification_steps:
    print(f"Step {step.step_number}: {step.title}")
    print(f"Formal code: {step.formal_code}")
```

### Benefits

- **Granular Control**: Formalize each verification step independently
- **Better Debugging**: Identify which steps fail during formalization
- **Modular Proofs**: Reuse common verification patterns
- **Clearer Structure**: Organized, readable verification process

## Examples

See the `examples/` directory for usage patterns:
- `examples/stepwise_demo.py` - Step-by-step verification demo

## Requirements

- Python 3.8+
- For HuggingFace models: `transformers`, `torch`
- For Ollama: `ollama` package and running Ollama server
- For configuration: `PyYAML`