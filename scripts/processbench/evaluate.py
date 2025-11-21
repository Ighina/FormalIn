from formalin.evaluation.metrics import parse_lean_file, postprocess_lean, processbench_f1, LeanCodeExtractor
from formalin.models import ModelFactory
import json
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def compute_metrics(evaluation_results: dict, prev_split: str, verbose: bool = False) -> dict:
    if verbose:
        print(f"Evaluating {prev_split} results...")
    
    step_f1 = f1_score(evaluation_results[prev_split]["step_labels"],
                        evaluation_results[prev_split]["step_predictions"])
    step_accuracy = accuracy_score(evaluation_results[prev_split]["step_labels"],
                                    evaluation_results[prev_split]["step_predictions"])
    step_precision = precision_score(evaluation_results[prev_split]["step_labels"],
                                    evaluation_results[prev_split]["step_predictions"])
    step_recall = recall_score(evaluation_results[prev_split]["step_labels"],
                                evaluation_results[prev_split]["step_predictions"])

    solution_f1 = f1_score(evaluation_results[prev_split]["solution_labels"],
                            evaluation_results[prev_split]["solution_predictions"])
    solution_accuracy = accuracy_score(evaluation_results[prev_split]["solution_labels"],
                                        evaluation_results[prev_split]["solution_predictions"])
    solution_precision = precision_score(evaluation_results[prev_split]["solution_labels"],
                                        evaluation_results[prev_split]["solution_predictions"])
    solution_recall = recall_score(evaluation_results[prev_split]["solution_labels"],
                                    evaluation_results[prev_split]["solution_predictions"])

    pb_f1, accuracy_incorrect, accuracy_correct = processbench_f1(evaluation_results[prev_split]["step_idx_actual"],
                                evaluation_results[prev_split]["step_idx_predicted"])
    
    if verbose:
        print(f"Step F1: {step_f1}")
        print(f"Step Accuracy: {step_accuracy}")
        print(f"Step Precision: {step_precision}")
        print(f"Step Recall: {step_recall}")
        print(f"Solution F1: {solution_f1}")
        print(f"Solution Accuracy: {solution_accuracy}")
        print(f"Solution Precision: {solution_precision}")
        print(f"Solution Recall: {solution_recall}")
        print(f"ProcessBench F1: {pb_f1}")

    evaluation_results[prev_split]["f1_step"] = step_f1
    evaluation_results[prev_split]["accuracy_step"] = step_accuracy
    evaluation_results[prev_split]["precision_step"] = step_precision
    evaluation_results[prev_split]["recall_step"] = step_recall
    evaluation_results[prev_split]["f1_solution"] = solution_f1
    evaluation_results[prev_split]["accuracy_solution"] = solution_accuracy
    evaluation_results[prev_split]["precision_solution"] = solution_precision
    evaluation_results[prev_split]["recall_solution"] = solution_recall
    evaluation_results[prev_split]["processbench_f1"] = pb_f1
    evaluation_results[prev_split]["accuracy_incorrect"] = accuracy_incorrect
    evaluation_results[prev_split]["accuracy_correct"] = accuracy_correct

    return evaluation_results

def main(lean_project: str, 
         result_file: str,
         extractor_model: str = None,
         clean: bool = True,
         verbose: bool = False) -> dict:
    
    with open(result_file, "r") as f:
        results = json.load(f)

    os.chdir(lean_project)

    if not os.path.exists(f"step_formalizations"):
        os.makedirs(
            f"step_formalizations",
            exist_ok=True,
        )

    prev_id = None
    prev_split = None
    whole_solution_correct = True
    step_idx = 0

    evaluation_results = {}

    extractor_class = None
    if extractor_model is not None:
        extractor_model = ModelFactory.create_provider(
            "vllm", # just vllm supported for this at the moment!
            extractor_model
        )
        extractor_class = LeanCodeExtractor(extractor_model)

    for problem in results:
        for result in problem["steps"]:
            if not whole_solution_correct:
                break

            split = result["metadata"]["split"]

            if evaluation_results.get(split) is None:
                if prev_split is not None:
                    evaluation_results = compute_metrics(evaluation_results, prev_split, verbose)
                
                evaluation_results[split] = {"step_predictions":[],
                                            "step_labels":[],
                                            "solution_predictions":[],
                                            "solution_labels":[],
                                            "step_idx_predicted":[],
                                            "step_idx_actual":[],
                                            "f1_step": None,
                                            "accuracy_step": None,
                                            "precision_step": None,
                                            "recall_step": None,
                                            "f1_solution": None,
                                            "accuracy_solution": None,
                                            "precision_solution": None,
                                            "recall_solution": None,
                                            "processbench_f1": None,
                                            "accuracy_incorrect": None,
                                            "accuracy_correct": None}
                prev_split = split

            problem_id = result["metadata"]["index"]
                
            label = result["metadata"]["first_incorrect"]
            evaluation_results[split]["step_labels"].append(int(step_idx < label))
            correct = result["metadata"]["correct"]

            if not os.path.exists(f"step_formalizations/{problem_id}"):

                os.makedirs(
                    f"step_formalizations/{problem_id}",
                    exist_ok=True,
                )

            
            if result["formal_proof"].strip():
                file_name = f"step_formalizations/{problem_id}/step_{step_idx}.lean"

                with open(file_name, "w") as f:
                    f.write(postprocess_lean(result["formal_proof"], extractor_class = extractor_class))

                parsed = parse_lean_file(file_path=file_name)

                if parsed:
                    evaluation_results[split]["step_predictions"].append(1)
                else:
                    evaluation_results[split]["step_predictions"].append(0)
                    whole_solution_correct = False

                if clean:
                    os.remove(file_name)
            else:
                evaluation_results[split]["step_predictions"].append(1)

            step_idx += 1
        
        evaluation_results[split]["step_idx_actual"].append(label)

        if whole_solution_correct:
            evaluation_results[split]["step_idx_predicted"].append(-1)
            evaluation_results[split]["solution_predictions"].append(1)
        else:
            evaluation_results[split]["step_idx_predicted"].append(step_idx - 1)
            evaluation_results[split]["solution_predictions"].append(0)
        evaluation_results[split]["solution_labels"].append(int(correct))
        step_idx = 0
        whole_solution_correct = True

    evaluation_results = compute_metrics(evaluation_results, prev_split, verbose)

    return evaluation_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lean_project", type=str, required=True, help="Path to the Lean project.")
    parser.add_argument("--result_file", type=str, required=True, help="Path to the JSON result file.")
    parser.add_argument("--extractor_model", type=str, required=False, help="If included, use the given LLM to extract Lean code.")
    parser.add_argument("--clean", action="store_true", help="Whether to clean up temporary files.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed evaluation info.")
    parser.add_argument("--save_metrics", type=str, default=None, help="Path to save the computed metrics as a JSON file.")

    args = parser.parse_args()

    metrics = main(
        lean_project=args.lean_project,
        result_file=args.result_file,
        extractor_model=args.extractor_model,
        clean=args.clean,
        verbose=args.verbose
    )

    print(json.dumps(metrics, indent=4))

    if args.save_metrics:
        with open(args.save_metrics, "w") as f:
            json.dump(metrics, f, indent=4)
