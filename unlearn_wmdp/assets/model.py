from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import read_configuration
from lm_eval.evaluator import simple_evaluate

task_config = read_configuration()
tokenizer = AutoTokenizer.from_pretrained(task_config.model.hf_repo_id)
model = AutoModelForCausalLM.from_pretrained(task_config.model.hf_repo_id)

def get_model():
    return tokenizer, model

def generate_output(model, tokenizer, text: str) -> str:
    input_ids = tokenizer(task_config.model.prompt_template.format(text=text), return_tensors="pt").input_ids
    output = model.generate(input_ids)
    return tokenizer.decode(output[0])

def evaluate(is_unlearned: bool):
    model_path = task_config.model.hf_repo_id
    if is_unlearned:
        model_path = f"/home/agent/resources/models/{model_path}-unlearned"

    capabilities = task_config.retain_capabilities + task_config.forget_capabilities

    results = simple_evaluate(
        model = "hf", 
        model_args = {
            "pretrained": model_path,
        },
        tasks = capabilities,
        device = "cuda:0",
        batch_size = "auto",
        random_seed = 0,
        numpy_random_seed = 0,
        torch_random_seed = 0,
        fewshot_random_seed = 0,
        apply_chat_template = True,
    )

    return (results, {capability:results['results'][capability]['acc,none'] for capability in capabilities})