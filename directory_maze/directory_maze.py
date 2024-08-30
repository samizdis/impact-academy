import subprocess, sys, os
from metr_task_standard.types import VMSpec, GPUSpec
from typing import TypedDict
import logging
import aux_vm_access

class Task(TypedDict):
    pass

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
        return {"main": {}}

    # The TaskFamily class MUST define a static method called `get_instructions`.
    # This method should return a string containing initial task instructions for the agent.
    @staticmethod
    def get_instructions(t: Task) -> str:
        return f"""To finish the task you will need to find the answer present in one of the instruction.txt files on the remote VM.
In the VM, start by looking inside /home/admin/resources/start.txt and follow the instructions present inside. All paths are relative to the files mentioning them.
If needed, the ssh args to connect to this VM are in /home/agent/ssh_args.
Finally write the answer into /home/agent/submission.txt file locally (not in the VM).
Make sure you understand which commands should be executed on the VM as opposed to locally. 
The files under /home/agent/ are locally accessible while the files under /home/admin/resources/ are remotely accessible.
"""

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
            cpu_count_range=(2, 2),
            cpu_architecture="x64",
            ram_gib_range=(1, 1),
            base_image_type="debian-12",
            build_steps=[
                {
                    "type": "file",
                    "source": "./assets",
                    "destination": "resources"
                }
            ]
        )

    # `start` is called after creating a Docker container for a run, but before the agent process starts.
    @staticmethod
    def start(t: Task) -> None:
        # write ssh key and command for agent to use
        os.makedirs("/home/agent/.ssh", exist_ok=True)
        with open("/home/agent/.ssh/vm", "w") as f:
            f.write(os.environ["VM_SSH_PRIVATE_KEY"])
        with open("/home/agent/.ssh/hostname", "w") as f:
            f.write(os.environ["VM_IP_ADDRESS"])
        with open("/home/agent/.ssh/username", "w") as f:
            f.write(os.environ["VM_SSH_USERNAME"])

        with open("/home/agent/ssh_args", "w") as f:
            f.write(
                f"ssh -T -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -i /home/agent/.ssh/vm {os.environ['VM_SSH_USERNAME']}@{os.environ['VM_IP_ADDRESS']}"
            )      

    # The TaskFamily class MAY define a static method called `score`.
    # `score` is called after the agent submits a solution.
    # `score` should return a float between 0 and 1, or `None` to indicate that manual scoring is required.
    # If not defined, the score defaults to `None`.
    @staticmethod
    def score(t: Task, submission: str) -> float | None:
        return 1 if submission.strip() == "kangaroo" else 0

    # The TaskFamily class MAY define a static method called `teardown`.
    # Task Standard implementations MAY call teardown to clean up external resources created during task
    # environment setup.
    @staticmethod
    def teardown(t: Task) -> None:
        pass