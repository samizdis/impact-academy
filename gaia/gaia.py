import logging
import subprocess
import sys
from typing import TypedDict
from variants import task_variants
import os


class Task(TypedDict):
    instructions: str
    dev_task_ids: dict
    test_task_ids: dict


logging.basicConfig(
    filename="/home/agent/logfile.log",
    filemode="a",
    level=logging.DEBUG,  # Minimum level of messages to log
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def run_command(cmd):
    print(f"Running: {cmd}", flush=True)
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        executable="/bin/bash"
    )
    exit_code = p.wait()
    if exit_code != 0:
        raise Exception(f"Command exited with non-zero exit code: {exit_code}")


# A task definition MUST define a TaskFamily class.
class TaskFamily:
    # The TaskFamily class MUST define a static string
    # called `standard_version` which declares which version
    # of the Task Standard this TaskFamily adheres to.
    standard_version = "0.2.3"

    required_environment_variables = [
        "HF_TOKEN",
        "OPENAI_API_KEY",
    ]

    @staticmethod
    def install() -> None:
        run_command("pip install huggingface_hub datasets python-dotenv")
        run_command("pip install langgraph langchain-openai langchain-experimental")

    @staticmethod
    def get_tasks() -> dict[str, Task]:
        return task_variants

    @staticmethod
    def get_instructions(t: Task) -> str:
        return t["instructions"]

    @staticmethod
    def get_permissions(t: Task) -> list[str]:
        return ["full_internet"]

    @staticmethod
    def get_aux_vm_spec(t: Task) -> None:
        return None

    # `start` is called after creating a Docker container for a run,
    # but before the agent process starts.
    @staticmethod
    def start(t: Task) -> None:
        run_command("mkdir -p /home/agent/resources/data")
        for _, dev_task in t["dev_task_ids"].items():
            f_name = dev_task["file_name"]
            if f_name:
                run_command(
                    (
                        f"cp data/gaia/validation/{f_name} "
                        f"/home/agent/resources/data/{f_name}"
                    )
                )
        run_command("cp assets/* /home/agent/resources/")
        run_command("cp ./.env /home/agent/resources/.env")
        hf_token = os.environ['HF_TOKEN'].strip()
        logging.info(
            f"Environment variable HF_TOKEN is {hf_token}"
        )
        if not hf_token:
            logging.error("Environment variable HF_TOKEN is not set. \
                Please set it to your huggingface token.")
        else:
            run_command(f"huggingface-cli login --token={hf_token}")
            run_command(f"runuser -l agent -c 'huggingface-cli login --token={hf_token}'")

    @staticmethod
    def score(t: Task, submission: str) -> float | None:
        return None

    @staticmethod
    def teardown(t: Task) -> None:
        pass
