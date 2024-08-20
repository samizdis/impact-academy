from typing import TypedDict
from instructions import default_instructions


class Task(TypedDict):
    variant: str
    instructions: str


task_variants: dict[str, Task] = {
    "default": {
        "variant": "default",
        "instructions": default_instructions,
    },
}