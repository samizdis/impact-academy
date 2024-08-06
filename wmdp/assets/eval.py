from lm_eval.evaluator import simple_evaluate
from utils import read_configuration
import argparse

task_config = read_configuration()

def evaluate(base_model):
    if not base_model:
        model_path = f"""models/{task_config["model"]["name"]}-{task_config["variant"]}"""
    else:
        model_path = f"""{task_config["model"]["hf_repo_id"]}"""

    capabilities = ["wmdp_cyber", "mmlu"]

    results = simple_evaluate(
        model = "hf", 
        model_args = {
            "pretrained": model_path
        },
        tasks = capabilities,
        device = "cuda:0",
        batch_size = "auto",
        limit=0.3,
        random_seed = 0,
        numpy_random_seed = 0,
        torch_random_seed = 0,
        fewshot_random_seed = 0
    )

    accuracies = {capability:results["results"][capability]["acc,none"] for capability in capabilities}

    print(f"Accuracies: {accuracies}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", action='store_true')
    args = parser.parse_args()
    results = evaluate(args.base_model)