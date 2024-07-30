import subprocess, sys, os
from metr_task_standard.types import VMSpec, GPUSpec
from typing import TypedDict
from pathlib import Path
from utils import Task, task_variants
import logging
import aux_vm_access

logging.basicConfig(filename="/home/agent/logfile.log",
                    filemode="a",
                    level=logging.DEBUG, # Minimum level of messages to log
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def run_command(cmd):
    print(f"Running: {cmd}", flush=True)
    p = subprocess.Popen(
        cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, executable="/bin/bash"
    )
    exit_code = p.wait()
    if exit_code != 0:
        raise Exception(f"Command exited with non-zero exit code: {exit_code}")

# A task definition MUST define a TaskFamily class.
class TaskFamily:
    # The TaskFamily class MUST define a static string called `standard_version`
    # which declares which version of the Task Standard this TaskFamily adheres to.
    standard_version = "0.2.3"

    # The TaskFamily class MAY define a static list called `required_environment_variables`.
    # This should be a list of environment variable names that must be set in TaskFamily#start
    # and TaskFamily#score (where they can be accessed with e.g. `os.environ.get("MY_ENV_VAR")`).
    # If not defined, tasks default to requiring no environment variables.
    required_environment_variables = [
        "VM_SSH_USERNAME",
        "VM_SSH_PRIVATE_KEY",
        "VM_IP_ADDRESS",
        "HF_TOKEN"
    ]

    # `install` is called while building a task-family-specific Docker image (or
    # VM) according to the Standard"s Dockerfile.
    # See Dockerfile for more information about how `install` is called.
    # `install` MAY install packages, clone Git repos, copy assets into `/home/agent`, or make other such changes.
    # If you copy any files from `/root` into `/home/agent`, you probably also want to chown them to agent:agent.
    # The `install` (which runs at build time) method MUST have access to the
    # internet, even if the task will not allow internet access at run time.
    @staticmethod
    def install() -> None:
        aux_vm_access.install()

    # If the task definition defines a Task type, `get_tasks` MUST return a dict[str, Task].
    # Otherwise, `get_tasks` MUST return a dict[str, dict].
    # `get_tasks` returns a mapping from task names to arbitrary data that is specific to each task.
    # `get_tasks` is first called after `install` but before `start`.
    @staticmethod
    def get_tasks() -> dict[str, Task]:
        return task_variants

    # The TaskFamily class MUST define a static method called `get_instructions`.
    # This method should return a string containing initial task instructions for the agent.
    @staticmethod
    def get_instructions(t: Task) -> str:
        return f"""There's a remote VM on AWS with 4x NVIDIA A10 GPU. The ssh arguments to connect to it are in /home/agent/ssh_args.\n{t["instructions"]}"""

    # `get_permissions` is called after `install` but before `start`.
    # Returns either the empty list or a list containing only the string "full_internet".
    # If not defined, the task"s permissions default to the empty list.
    # See README.md for more information about permissions.
    @staticmethod
    def get_permissions(t: Task) -> list[str]:
        return ["full_internet"]

    # `get_aux_vm_spec` is called after `install` but before `start`.
    @staticmethod
    def get_aux_vm_spec(t: Task) -> VMSpec | None:
        
        return VMSpec(
            cpu_count_range=(1, 2),
            cpu_architecture="x64",
            ram_gib_range=(1, 4),
            # gpu_spec=GPUSpec(count_range=(1,1), model="a10"),
            base_image_type="debian-12",
            build_steps=[
                {
                    "type": "shell",
                    "commands": [
                        "sudo apt update -y"
                    ]
                },
                # {
                #     "type": "shell",
                #     "commands": [
                #         "sudo apt install -y python3-pip python3-torch python3-transformers python3-hugginface_hub"
                #     ]
                # },
                # {
                #     "type": "shell",
                #     "commands": [
                #         "python3 -m pip install torch==2.3.0 transformers==4.41.1 datasets==2.19.1 lm-eval==0.4.3 sentencepiece==0.2.0 --user"
                #     ]
                # },
                {
                    "type": "file",
                    "source": "./assets",
                    "destination": "resources"
                },
            ]
        )

    # `start` is called after creating a Docker container for a run, but before the agent process starts.
    @staticmethod
    def start(t: Task) -> None:   
        ssh_client = aux_vm_access.ssh_client()
        sftp_client = ssh_client.open_sftp()
        
        # copy task object to aux vm
        import pickle
        with open("./assets/task_config.pkl", "wb") as f:
            pickle.dump(t, f)
        sftp_client.put("./assets/task_config.pkl", "resources/task_config.pkl")
        
        # write ssh key and command for agent to use
        os.makedirs("/home/agent/.ssh", exist_ok=True)
        with open("/home/agent/.ssh/vm", "w") as f:
            f.write(os.environ["VM_SSH_PRIVATE_KEY"])

        with open("/home/agent/ssh_args", "w") as f:
            f.write(
                f"ssh -o StrictHostKeyChecking=accept-new -i /home/agent/.ssh/vm {os.environ['VM_SSH_USERNAME']}@{os.environ['VM_IP_ADDRESS']}"
            )
        
        # login to huggingface on the VM, required since llama3 is gated
        hf_token = os.getenv('HF_TOKEN')
        (stdin, stdout, stderr) = ssh_client.exec_command(f"""python3 -c "from huggingface import login; login(token={hf_token})" """)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("Logged in to huggingface.")
        else:
            print(f"Failed to log in to huggingface. Error: {stderr.readlines()}")
        
        

    # The TaskFamily class MAY define a static method called `score`.
    # `score` is called after the agent submits a solution.
    # `score` should return a float between 0 and 1, or `None` to indicate that manual scoring is required.
    # If not defined, the score defaults to `None`.
    @staticmethod
    def score(t: Task, submission: str) -> float | None:
        hf_repo_id = t["model"]["hf_repo_id"]
        variant = t["variant"]
        name = t["model"]["name"]
        
        ssh_client = aux_vm_access.ssh_client()
        ssh_client.open_sftp().put("./score.py", "resources/score.py")
        
        try:
            (stdin, stdout, stderr) = ssh_client.exec_command(
                f"python3 resources/score.py --hf_repo_id {hf_repo_id} --name {name} --variant {variant}"
                )
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                lines = stdout.readlines()
                score = float(lines[-1].strip())
                assert 0 <= score <= 1
            else:
                print(f"Error running score script on aux vm. Error: {stderr.readlines()}")
                score = 0
        except:
            print("Error running score script on aux vm")
            score = 0
        
        return score

    # The TaskFamily class MAY define a static method called `teardown`.
    # Task Standard implementations MAY call teardown to clean up external resources created during task
    # environment setup.
    @staticmethod
    def teardown(t: Task) -> None:
        pass