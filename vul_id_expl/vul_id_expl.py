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
        run_command("cp ./.env /home/agent/.env")

        # CSE2 Vulnerability ID and exploitation-specific setup
        run_command("git clone https://github.com/meta-llama/PurpleLlama.git /home/agent/PurpleLlama")
        run_command("pip3 install -r /home/agent/PurpleLlama/CybersecurityBenchmarks/requirements.txt")
        run_command("export DATASETS=$/home/agent/PurpleLlama/CybersecurityBenchmarks/datasets")
        run_command(f"export OPENAI_API_KEY={os.environ['OPENAI_API_KEY']}")

        logging.info(f"{os.environ['OPENAI_API_KEY']}")
        run_command(f"echo -e \"export OPENAI_API_KEY=\"{os.environ['OPENAI_API_KEY']}\"\" >> ~/.bashrc")

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
        logging.info("CURRENT WD", os.getcwd())

        DATA = "/home/agent/PurpleLlama/CybersecurityBenchmarks/datasets"

        command = f"python3 -m PurpleLlama.CybersecurityBenchmarks.benchmark.run " \
          f"--benchmark=\"canary-exploit\" " \
          f"--prompt-path=\"/home/agent/resources/canary_exploit/training.json\" " \
          f"--response-path=\"{DATA}/canary_exploit/canary_exploit_responses.json\" " \
          f"--judge-response-path=\"{DATA}/canary_exploit/canary_exploit_judge_responses.json\" " \
          f"--stat-path=\"{DATA}/canary_exploit/canary_exploit_stat.json\" " \
          f"--llm-under-test=OPENAI::gpt-4o::{os.environ['OPENAI_API_KEY']}"

        #subprocess.call(command, shell=True)
        

    @staticmethod
    def start(t: Task) -> None:

        run_command("cp -r /root/resources /home/agent/resources")
        run_command("chown -R agent:agent /home/agent/resources")
        run_command("chmod +x /home/agent/resources/canary_exploit/eval.sh")

        import os
        import tempfile
        from pathlib import Path
        import importlib.resources
        import shutil

        def _get_semgrep_core_path() -> Path:
            semgrep_core_name = "semgrep-core"
            try:
                with importlib.resources.path("semgrep.bin", semgrep_core_name) as path:
                    if path.is_file():
                        return path
            except (FileNotFoundError, ModuleNotFoundError):
                pass

            path = shutil.which(semgrep_core_name)
            if path is not None:
                return Path(path)

            raise Exception(
                f"Failed to find {semgrep_core_name} in PATH or in the semgrep package."
            )
        
        def make_semgrep_binary_path():
            source = "/usr/local/lib/python3.11/site-packages/semgrep/bin/semgrep-core"
            # Use a temporary directory that the user has write access to
            temp_dir = tempfile.gettempdir()
            destination = os.path.join(temp_dir, "osemgrep")
            os.symlink(source, destination)
            return Path(destination)

        source = _get_semgrep_core_path()
        destination = Path(str(source).replace("semgrep-core", "osemgrep"))
        os.symlink(source, destination)
        


    @staticmethod
    def score(t: Task, submission: str) -> float | None:

        from dotenv import load_dotenv

        load_dotenv()


        return 0

    @staticmethod
    def teardown(t: Task) -> None:
        pass
