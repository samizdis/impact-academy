from typing import TypedDict
from instructions import (
    default_instructions,
    fixed_traj_path_instructions,
    less_handholding_instructions,
)


class Task(TypedDict):
    variant: str
    instructions: str


task_variants: dict[str, Task] = {
    "default": {
        "variant": "default",
        "instructions": default_instructions,
    },
    "fixed_traj_path": {
        "variant": "default",
        "instructions": fixed_traj_path_instructions,
    },
    "less_handholding_instructions": {
        "variant": "default",
        "instructions": less_handholding_instructions,
    },
}
