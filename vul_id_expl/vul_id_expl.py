from metr_task_standard.types import VMSpec
from typing import TypedDict
import logging
import sys, os, subprocess, json
from datetime import datetime

logging.basicConfig(
    filename="logfile_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
    filemode="a",
    level=logging.INFO,  # Minimum level of messages to log
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def run_command(cmd):
    logging.info(f"Running: {cmd}")
    p = subprocess.Popen(
        cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, executable="/bin/bash"
    )
    exit_code = p.wait()
    if exit_code != 0:
        raise Exception(f"Command exited with non-zero exit code: {exit_code}")


class Task(TypedDict):
    variant: str


class TaskFamily:
    standard_version = "0.2.3"

    required_environment_variables = ["OPENAI_API_KEY"]

    @staticmethod
    def install() -> None:
        run_command(f"pip install openai python-dotenv")

        # CSE2 Vulnerability ID and exploitation-specific setup
        run_command("git clone https://github.com/meta-llama/PurpleLlama.git")
        run_command("pip3 install -r /root/PurpleLlama/CybersecurityBenchmarks/requirements.txt")



    @staticmethod
    def get_tasks() -> dict[str, Task]:
        return {
            "main": {
                "model_name": "gpt-4o",
                "instructions_file_name": "instructions.txt",
                "max_steps": 30
            }
        }

    @staticmethod
    def get_instructions(t: Task) -> str:
        with open("{}".format(t["instructions_file_name"])) as f:
            instructions = f.read()
        return instructions

    @staticmethod
    def get_permissions(t: Task) -> list[str]:
        return ["full_internet"]

    @staticmethod
    def get_aux_vm_spec(t: Task) -> VMSpec | None:
        return None

    @staticmethod
    def run_evaluation(t: Task)-> None:
        from resources.canary_exploit import canary_exploit_benchmark
        DATASET = "/root/canary_exploit"
        command = (
                f"python3 canary_exploit.run "
                f"benchmark={DATASET} "
                f"prompt-path={DATASET}.json "
                f"response-path={DATASET}/canary_exploit_responses.json "
                f"judge-response-path={DATASET}/canary_exploit_judge_responses.json "
                f"stat-path={DATASET}/canary_exploit_stat.json "
                f"llm-under-test=OPENAI::gpt-4o::{os.environ['OPENAI_API_KEY']}"
                )

        subprocess.call(command, shell=True)
        

    @staticmethod
    def start(t: Task) -> None:

        with open("resources/training_prompts.json", "r") as f:
            prompts = json.load(f)
            
        TaskFamily.run_evaluation()        



    @staticmethod
    def score(t: Task, submission: str) -> float | None:

        from dotenv import load_dotenv

        load_dotenv()


        return 0

    @staticmethod
    def teardown(t: Task) -> None:
        pass
