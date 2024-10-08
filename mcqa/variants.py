from typing import TypedDict
from instructions import (
    mmlu_instructions,
)


class Task(TypedDict):
    variant: str
    instructions: str


task_variants: dict[str, Task] = {
    "mmlu": {
        "variant": "mmlu",
        "instructions": mmlu_instructions,
    },
}
