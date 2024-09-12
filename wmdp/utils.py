from typing import TypedDict
from instruction_templates import unlearning_instructions_hard_template, unlearning_instructions_easy_template

class Task(TypedDict):
    instructions: str
    model: dict = None
    variant: str

llama3_8b = {
    'name':"Meta-Llama-3-8B", 
    'hf_repo_id':"meta-llama/Meta-Llama-3-8B", 
}

smol_lm_360m = {
    'name':"SmolLM-360M",
    'hf_repo_id':"HuggingFaceTB/SmolLM-360M",
}

wmdp = "wmdp"
commonsense_qa_bm = "commonsense_qa"
mmlu_bm = "mmlu"


task_variants : dict[str, Task] = {
    "unlearn_main_hard": {
        "model": llama3_8b,
        "instructions": unlearning_instructions_hard_template.format(model_name=llama3_8b['name'], hf_repo_id=llama3_8b['hf_repo_id']),
        "variant": "unlearn",
        "difficulty": "hard"
    },
    
    "unlearn_main_easy": {
        "model": llama3_8b,
        "instructions": unlearning_instructions_easy_template.format(model_name=llama3_8b['name'], hf_repo_id=llama3_8b['hf_repo_id']),
        "variant": "unlearn",
        "difficulty": "easy"
    }
    
}
