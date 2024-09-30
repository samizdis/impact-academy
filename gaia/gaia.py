import subprocess
import sys
import logging
from typing import TypedDict


class Task(TypedDict):
    instructions: str


task_variants = {
    "main": {
        "instructions": "",
        "dev_task_ids": {
            "Game Show Puzzle": "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",
            "Secret Santa": "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb",
            "Bags of Potatoes": "dc28cf18-6431-458b-83ef-64b3ce566c10",
            "Spreadsheet Sales": "7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f",
            "Seagull": "50ad0280-0819-4bd9-b275-5de32d3b5bcb",
            "Guava vs Pineapple": "4b650a35-8529-4695-89ed-8dc7a500a498",
        },
        "test_task_ids": {
            "Gold Coin Adventure": "6af95c8f-8cbf-4c12-b02c-f9a23cc1ecb9",
            "Excel Locomotive": "cfd773c8-371f-425c-b081-f254f96c0530",
            "Precipitation CSV": "7805912b-c8da-4134-9b54-b590f884352a",
            "Find the Line": "f1ba834a-3bcb-4e55-836c-06cc1e2ccb9f",
            "Integer Fun": "314b3c26-3e8d-4f8c-a9aa-c77275072e0b",
            "Twinkle Twinkle": "68ccf11a-bcd3-41e5-a5ee-3e29253449e9",
        }
    }}

logging.basicConfig(
    filename="/home/agent/logfile.log",
    filemode="a",
    level=logging.DEBUG,  # Minimum level of messages to log
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def run_command(cmd):
    print(f"Running: {cmd}", flush=True)
    p = subprocess.Popen(
        cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr,
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
        pass

    @staticmethod
    def score(t: Task, submission: str) -> float | None:
        return None

    @staticmethod
    def teardown(t: Task) -> None:
        pass
