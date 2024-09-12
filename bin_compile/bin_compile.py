from metr_task_standard.types import VMSpec
from typing import TypedDict
import json
import logging
import subprocess, sys, os


logging.basicConfig(filename='logfile.log',
                    filemode='a',
                    level=logging.DEBUG, # Minimum level of messages to log
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# If you try to import third-party libraries here, the task will crash because `install` hasn't run yet.
# You can import libraries in `start` and `score`, which run after `install`.

# The Task class contains data specific to each task in the family.
class Task(TypedDict):
    task_id: int
    compilation_type: str
    purpose: str
    c_func: str
    c_test: str
    input_asm_prompt: str

SYSTEM_INSTRUCTIONS = "system_instructions.txt"


def run_command(cmd):
    print(f"Running: {cmd}", flush=True)
    p = subprocess.Popen(
        cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, executable="/bin/bash"
    )
    exit_code = p.wait()
    if exit_code != 0:
        raise Exception(f"Command exited with non-zero exit code: {exit_code}")

# A breakdown of what we are going for
# We implement the direct compilation process
# We need to have a mapping between binary files and the purpose of the function in those assembled files
# Setup
# Post setup the following happens
# We give instructions to the TLA to start upliftment of the baseagent
# What actions can the baseagent perform? - 
# What actions can the TLA perform? - can it change the scaffold code of the baseline agent? 

class TaskFamily:
    # Which version of the Task Standard this TaskFamily adheres to
    standard_version = "0.2.2"

    # Optional: a list of environment variable names that must be set in TaskFamily#start
    # and TaskFamily#score (where they can be accessed with e.g. `os.environ.get("MY_ENV_VAR")`).
    required_environment_variables = ["OPENAI_API_KEY"]

    # `install` is called while building a task-family-specific Docker image.
    # See the Dockerfile for more information about how `install` is called.
    # `install` may install packages, clone Git repos, copy assets into `/home/agent`, or make other such changes.
    # The `install` method (which runs at build time) can access the
    # internet, even if the task will not allow internet access at run time.
    @staticmethod
    def install() -> None:
        pass

    # `get_tasks` returns a mapping from task names to arbitrary data that is specific to each task.
    # `get_tasks` is first called after `install` but before `start`.
    # Note that this method should return the same value every time it is called.
    @staticmethod
    def get_tasks() -> dict[str, Task]:
        with open("decompile-eval-executable-gcc-obj.json") as f:
            dataset = json.load(f)
        index = 0 # TODO: Implement a policy to construct and output a "suite"
        return { "main": 
            { "asm": dataset[index]["input_asm_prompt"], "compilation_type": dataset[index]["compilation_type"], "purpose": dataset[index]["purpose"] }  
        }     

    # `get_instructions` returns a string containing initial task instructions for the agent.
    @staticmethod
    def get_instructions(t: Task) -> str:
        # problem = t["problem"]
        # return f"""Solve the following math problem: {problem}"""

        with open("{}".format(SYSTEM_INSTRUCTIONS)) as f:
            instructions = f.read()
        insturctions = instructions + "\nThe purpose of the assembled code is {}\nThe compilation type is {}\n".format(
            t['purpose'],
            t['compilation_type']
        )
        logging.debug(instructions)
        return instructions
        

    # `get_permissions` is called after `install` but before `start`.
    # It returns either the empty list or a list containing only the string "full_internet".
    # Refer to the Task Standard more information.
    @staticmethod
    def get_permissions(t: Task) -> list[str]:
        return ["full_internet"]

    # Optional: `get_aux_vm_spec` specifies an auxiliary virtual machine that will be set up along with the task.
    # This is useful if you want to let the agent use a GPU.
    # Refer to the Task Standard for more information.
    @staticmethod
    def get_aux_vm_spec(t: Task) -> VMSpec | None:
        return None

    # `start` is called after creating a Docker container for a run, but before the agent process starts.
    # `start` may copy task-specific assets into /home/agent, start long-running processes like web servers,
    # and set up external services like additional cloud servers.
    @staticmethod
    def start(t: Task) -> None:
        run_command(f"cp -a resources/. /home/agent/resources/")
        run_command(f"cp -a utils/. /home/agent/utils")

    # `score` is called after the agent submits a solution. If the solution is more complex than a short string,
    # `score` can read files, run programs, make network requests, etc.
    # `score` should return a float between 0 and 1, or `None` to indicate that manual scoring is required.
    @staticmethod
    def score(t: Task, submission: str) -> float | None:
        # Make sure to handle errors if the agent submits an invalid solution
        print("scoring...")
        print(submission)
        run_command(f"sudo apt --version")
        run_command(f"gcc --version")
        try:
            answer_int = int(submission)
        except ValueError:
            print("Answer must be an integer")
            return 0

        if answer_int == t["answer"]:
            return 1
        else:
            return 0
        
    # Optional: `teardown` cleans up any external resources created during setup.
    @staticmethod
    def teardown(t: Task) -> None:
        pass
