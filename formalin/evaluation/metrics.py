"""
Evaluation metrics for model answers.
"""

from typing import List, Union, Dict
from collections import Counter
import json
import numpy as np
import os
import re

import subprocess
import sys


def simple_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate simple accuracy between predictions and ground truths.

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers

    Returns:
        Accuracy as a float between 0 and 1

    Raises:
        ValueError: If predictions and ground_truths have different lengths
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for pred, truth in zip(predictions, ground_truths) if pred == truth)
    return correct / len(predictions)


def majority_vote_accuracy(
    prediction_sets: List[List[str]], ground_truths: List[str]
) -> float:
    """
    Calculate accuracy using majority vote from multiple predictions per sample.

    Args:
        prediction_sets: List of prediction lists, where each inner list contains
                        multiple predictions for a single sample
        ground_truths: List of ground truth answers

    Returns:
        Accuracy as a float between 0 and 1

    Raises:
        ValueError: If prediction_sets and ground_truths have different lengths
    """
    if len(prediction_sets) != len(ground_truths):
        raise ValueError("Prediction sets and ground truths must have the same length")

    if len(prediction_sets) == 0:
        return 0.0

    majority_predictions = []

    for predictions in prediction_sets:
        if not predictions:
            majority_predictions.append("")
            continue

        # Count occurrences of each prediction
        counter = Counter(predictions)
        # Get the most common prediction
        majority_pred = counter.most_common(1)[0][0]
        majority_predictions.append(majority_pred)

    return simple_accuracy(majority_predictions, ground_truths)


def weighted_vote_accuracy(
    prediction_sets: List[List[str]],
    weights_sets: List[List[float]],
    ground_truths: List[str],
) -> float:
    """
    Calculate accuracy using weighted votes from multiple predictions per sample.

    The weight replaces the actual count of answers. For example, if "answer 1"
    appears 2 times with weight 0.5, it will count as 1 (2 * 0.5).

    Args:
        prediction_sets: List of prediction lists, where each inner list contains
                        multiple predictions for a single sample
        weights_sets: List of weight lists, where each inner list contains
                     weights corresponding to the predictions
        ground_truths: List of ground truth answers

    Returns:
        Accuracy as a float between 0 and 1

    Raises:
        ValueError: If input lists have mismatched lengths or if predictions and
                   weights don't match within each set
    """
    if len(prediction_sets) != len(ground_truths):
        raise ValueError("Prediction sets and ground truths must have the same length")

    if len(prediction_sets) != len(weights_sets):
        raise ValueError("Prediction sets and weights sets must have the same length")

    if len(prediction_sets) == 0:
        return 0.0

    weighted_predictions = []

    for predictions, weights in zip(prediction_sets, weights_sets):
        if len(predictions) != len(weights):
            raise ValueError(
                "Predictions and weights must have the same length for each sample"
            )

        if not predictions:
            weighted_predictions.append("")
            continue

        # Calculate weighted scores for each unique prediction
        weighted_scores = {}
        for pred, weight in zip(predictions, weights):
            if pred not in weighted_scores:
                weighted_scores[pred] = 0.0
            weighted_scores[pred] += weight

        # Get the prediction with the highest weighted score
        if weighted_scores:
            best_pred = max(weighted_scores, key=weighted_scores.get)
            weighted_predictions.append(best_pred)
        else:
            weighted_predictions.append("")

    return simple_accuracy(weighted_predictions, ground_truths)

def processbench_f1(y_true, y_pred) -> float:
    error_data = [int(y==y_pred[idx]) for idx, y in enumerate(y_true) if y != -1]
    correct_data = [int(y==y_pred[idx]) for idx, y in enumerate(y_true) if y == 1]

    acc_incorrect = np.mean([e for e in error_data]) * 100
    acc_correct = np.mean([e for e in correct_data]) * 100
    f1 = 2 * acc_incorrect * acc_correct / (acc_incorrect + acc_correct)

    return f1, acc_incorrect, acc_correct

def parse_lean_file(file_path):
    """
    Parse a Lean file using lake env lean.
    Returns True if successful (no errors, only warnings allowed), False otherwise.
    """
    try:
        # Run lean via lake env
        result = subprocess.run(
            ["lake", "env", "lean", file_path],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
        )

        # Check if there's any output
        output = result.stdout + result.stderr

        # If there's output, check if it contains errors
        if output.strip():
            # Check for error indicators
            # Lean errors typically contain "error:" in the output
            if "error:" in output.lower():
                print(f"Error found in {file_path}")
                return False
            # If only warnings, it's still OK
            elif "warning:" in output.lower():
                print(f"Warnings only in {file_path} (OK)")
                return True
            else:
                # Any other non-empty output might indicate an issue
                print(f"Unexpected output from {file_path}")
                return False

        # No output means success
        return True

    except subprocess.TimeoutExpired:
        print(f"Timeout while parsing {file_path}")
        return False
    except FileNotFoundError:
        print(
            "Error: 'lake' command not found. Make sure you're in a Lean project directory."
        )
        return False
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return False


def postprocess_lean(lean_code: str, retrieve_first: bool = True) -> str:
    """
    Postprocess Lean code to avoid common errors

    Args:
        lean_code: the input lean code as a string

    Returns:
        the postprocessed lean code
    """
    # 1. Remove all occurrences of triple backticks
    if retrieve_first:
        try:
            lean_code = re.findall("```lean(.+?)```", lean_code, flags=re.DOTALL)[-1]
        except IndexError:
            print(lean_code)
            print("INCOMPLETE!")
            
    processed = lean_code.replace("```", "")

    # 2. Remove "lean" at the very beginning (strip leading spaces first)
    processed = processed.lstrip()
    if processed.startswith("lean4"):
        processed = processed[len("lean4") :].lstrip()
    elif processed.startswith("lean"):
        processed = processed[len("lean") :].lstrip()

    # 3. Remove lines starting with "import "
    lines = processed.splitlines()
    lines = [
        line
        for line in lines
        if not line.strip().startswith("import ") or line.startswith("open")
    ]

    # 4. Ensure string starts with "import Mathlib"
    processed = "\n".join(lines).lstrip()
    if not processed.startswith("import Mathlib"):
        processed = "import Mathlib\n" + processed

    return processed


def lean_compile_success(
    result_file: str, lean_project: str, clean: bool = True
) -> Dict[str, float]:
    """
    Calculate the number of successes in parsing generated lean code, both at the step level
    and at the general level.

    Args:
        result_file: Path to the JSON file containing model results. The file should contain a list of dictionaries,
                     each with keys "verification_steps" and "formal_code" inside the verification_steps.

        lean_project: Path to the Lean project where to write the lean files. Should point to a valid Lean project.

        clean: if True, delete the file straight after having attempted compilation with lean.

    Returns:
        Dictionary of success rates with keys:
            - "step_level_success": Proportion of steps that compiled successfully
            - "general_level_success": Proportion of samples having all steps compiling successfully

    Raises:
        FileNotFoundError: If input file/lean project path does not exist.
    """

    with open(result_file) as f:
        results = json.load(f)

    os.chdir(lean_project)

    if not os.path.exists(f"step_formalizations"):
        os.makedirs(
            f"step_formalizations",
            exist_ok=True,
        )

    prev_id = None
    whole_solution_correct = False
    solutions_counter = 0
    correct_solution_counter = 0
    total_steps = 0
    correct_steps = 0

    for result in results:
        total_steps += 1
        unique_id = {result["metadata"]["unique_id"]}
        if unique_id != prev_id:
            solutions_counter += 1

            if whole_solution_correct:
                correct_solution_counter += 1

        if not os.path.exists(f"step_formalizations/{unique_id}"):

            os.makedirs(
                f"step_formalizations/{unique_id}",
                exist_ok=True,
            )

        for idx, step in enumerate(result["verification_steps"]):
            file_name = f"step_formalizations/{unique_id}/step_{idx}.lean"

            with open(file_name, "w") as f:
                f.write(postprocess_lean(step["formal_code"]))

            parsed = parse_lean_file(file_path=file_name)

            if parsed:
                print("CORRECT!")
                correct_steps += 1
                if unique_id != prev_id:
                    whole_solution_correct = True
            else:
                whole_solution_correct = False

            if clean:
                os.remove(file_name)

    if whole_solution_correct:
        correct_solution_counter += 1

    return {
        "step_level_success": correct_steps / total_steps,
        "general_level_success": correct_solution_counter / solutions_counter,
    }
