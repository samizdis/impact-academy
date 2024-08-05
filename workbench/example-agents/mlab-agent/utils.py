import openai
import anthropic
import tiktoken

import dataclasses
from dataclasses import dataclass
from argparse import Namespace
import json
from typing import Any, Dict, List

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        #if it is a function, use its string name
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif hasattr(o, '__call__'):
            return o.__name__
        elif isinstance(o, Namespace):
            return vars(o)

        return super().default(o)

class TooLongPromptError(Exception):
    pass
class LLMError(Exception):
    pass

class EnvException(Exception):
    def __init__(self, message):
        self.message = message 
    def __str__(self):
        return self.message

@dataclass(frozen=True)
class ActionInfo:
    name: str
    description: str
    usage: dict
    return_value: str
    function: str
    is_primitive: bool = False

@dataclass(frozen=True)
class Action:
    name: str
    args: Dict[str, Any]


@dataclass(frozen=True)
class Step:
    action: Action
    observation: str  # What was returned
    timestamp: float  # When the action was taken


@dataclass(frozen=True)
class Trace:
    steps: List[Step]
    low_level_steps: List[Step]
    action_infos: Dict[str, ActionInfo]
    task_description: str


def complete_text_openai(prompt, stop_sequences=["Observation:"], model="gpt-3.5-turbo", max_tokens_to_sample=500, temperature=0.2, log_file=None, **kwargs):
        """ Call the OpenAI API to complete a prompt."""
        raw_request = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens_to_sample,
            "stop": stop_sequences or None,  # API doesn't like empty list
            **kwargs
        }
        if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(**{"messages": messages,**raw_request})
            completion = response["choices"][0]["message"]["content"]
        else:
            response = openai.Completion.create(**{"prompt": prompt,**raw_request})
            completion = response["choices"][0]["text"]
        if log_file is not None:
            log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
        return completion
    
def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    enc = tiktoken.get_encoding("cl100k_base")

    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")