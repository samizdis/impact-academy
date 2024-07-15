
import dataclasses
from typing import TypedDict

@dataclasses.dataclass
class Dataset:
    path: str
    split: str
    name: str = None

@dataclasses.dataclass
class Benchmark:
    name: str

@dataclasses.dataclass
class Model:
    name: str
    hf_repo_id: str
    prompt_template: str = None

class Task(TypedDict):
    retain_capabilities: list[str]
    forget_capabilities: list[str]
    model: Model

llama_uncensored = Model(
    name="llama-7b-uncensored", 
    hf_repo_id="georgesung/llama2_7b_chat_uncensored", 
    prompt_template="""### Human:\n{text}\n\n### Response:\n"""
)

wmdp = "wmdp"
commonsense_qa_bm = "commonsense_qa"
mmlu_bm = "mmlu"


task_variants : dict[str, Task] = {
    "cyber" : {
        "model": llama_uncensored, 
        "forget_capabilities": [wmdp],
        "retain_capabilities": [commonsense_qa_bm, mmlu_bm]
    },
    "bio" : {
        "model": llama_uncensored, 
        "forget_capabilities": [wmdp],
        "retain_capabilities": [commonsense_qa_bm, mmlu_bm]
    },
    "chem" : {
        "model": llama_uncensored, 
        "forget_capabilities": [wmdp],
        "retain_capabilities": [commonsense_qa_bm, mmlu_bm]
    }
}
