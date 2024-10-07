import logging
import subprocess
import sys
from typing import TypedDict
from variants import task_variants


class Task(TypedDict):
    instructions: str


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

    required_environment_variables = []

    @staticmethod
    def install() -> None:
        pass

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
        for dev_task in t["dev_task_ids"].items():
            f_name = dev_task["file_name"]
            run_command(
                (
                    f"cp data/gaia/validation/{f_name} "
                    f"/home/agent/resources/data/{f_name}"
                )
            )
        run_command("cp assets/* /home/agent/resources/")

    @staticmethod
    def score(t: Task, submission: str) -> float | None:
        return None

    @staticmethod
    def teardown(t: Task) -> None:
        pass
