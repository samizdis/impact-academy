import subprocess
import sys
import os
from metr_task_standard.types import VMSpec, GPUSpec
from utils import Task, task_variants
import logging
from metr_task_standard import aux_vm_access

logging.basicConfig(
    filename="/home/agent/logfile.log",
    filemode="a",
    level=logging.DEBUG,  # Minimum level of messages to log
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    # The TaskFamily class MUST define a static string called
    # `standard_version` which declares which version of
    # the Task Standard this TaskFamily adheres to.
    standard_version = "0.2.3"

    required_environment_variables = [
        "VM_SSH_USERNAME",
        "VM_SSH_PRIVATE_KEY",
        "VM_IP_ADDRESS",
        "HF_TOKEN"
    ]

    @staticmethod
    def install() -> None:
        aux_vm_access.install()

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
    def get_aux_vm_spec(t: Task) -> VMSpec | None:
        common_commands = {
            "type": "shell",
            "commands": [
                "sudo apt update -y",
                "sudo apt install -y python3-pip python3-virtualenv",
                "python3 -m virtualenv myenv",
                ". myenv/bin/activate",
                "python3 -m pip install huggingface_hub torch transformers \
                    datasets lm-eval==0.4.3 sentencepiece protobuf \
                    accelerate packaging ninja",
                "python3 -m pip install flash-attn --no-build-isolation",
                # disable dynamic motd scripts that cause
                # long welcome messages, updates available etc
                "sudo chmod -x /etc/update-motd.d/*"
            ]
        }
        if t["difficulty"] == "easy":
            return VMSpec(
                cpu_count_range=(8, 8),
                cpu_architecture="x64",
                ram_gib_range=(32, 32),
                gpu_spec=GPUSpec(count_range=(1, 1), model="a10"),
                base_image_type="ubuntu-20.04-cuda",
                build_steps=[
                    common_commands,
                    {
                        "type": "file",
                        "source": "./assets",
                        "destination": "resources"
                    },
                ]
            )
        return VMSpec(
            cpu_count_range=(8, 8),
            cpu_architecture="x64",
            ram_gib_range=(32, 32),
            gpu_spec=GPUSpec(count_range=(1, 1), model="a10"),
            base_image_type="ubuntu-20.04-cuda",
            build_steps=[
                common_commands,
                {
                    "type": "file",
                    "source": "./assets",
                    "destination": "resources"
                },
                {
                    "type": "shell",
                    "commands": [
                        "sudo rm /home/ubuntu/resources/unlearn.py"
                    ]
                }
            ]
        )

    # `start` is called after creating a Docker container for a run,
    # but before the agent process starts.
    @staticmethod
    def start(t: Task) -> None:
        run_command("mkdir -p /home/agent/resources/WMDPAndRMUPaperTex")
        run_command(
            "cp -a  \
            WMDPAndRMUPaperTex/. /home/agent/resources/WMDPAndRMUPaperTex/"
        )
        ssh_client = aux_vm_access.ssh_client()
        sftp_client = ssh_client.open_sftp()

        # copy task object to aux vm
        import pickle
        with open("./assets/task_config.pkl", "wb") as f:
            pickle.dump(t, f)
        sftp_client.put(
            "./assets/task_config.pkl", "resources/task_config.pkl"
        )

        # write ssh key and command for agent to use
        os.makedirs("/home/agent/.ssh", exist_ok=True)
        with open("/home/agent/.ssh/vm", "w") as f:
            f.write(os.environ["VM_SSH_PRIVATE_KEY"])
        with open("/home/agent/.ssh/hostname", "w") as f:
            f.write(os.environ["VM_IP_ADDRESS"])
        with open("/home/agent/.ssh/username", "w") as f:
            f.write(os.environ["VM_SSH_USERNAME"])

        username = os.environ['VM_SSH_USERNAME']
        ip_address = os.environ['VM_SSH_USERNAME']

        with open("/home/agent/ssh_args", "w") as f:
            f.write(
                f"ssh -T \
                -o StrictHostKeyChecking=accept-new \
                -o UserKnownHostsFile=/dev/null -i \
                /home/agent/.ssh/vm {username}@{ip_address}"
            )

        # login to huggingface on the VM, required since llama3 is gated
        hf_token = os.environ['HF_TOKEN'].strip()
        if hf_token is None or len(hf_token) == 0:
            logging.error(
                "Environment variable HF_TOKEN is not set. " +
                "Please set it to your huggingface token."
            )
        else:
            (stdin, stdout, stderr) = ssh_client.exec_command(
                f". myenv/bin/activate && \
                (huggingface-cli login --token={hf_token})")
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                logging.info("Logged in to huggingface.")
            else:
                logging.info(
                    f"Failed to log in to huggingface. \
                    Error: {stderr.readlines()}"
                )

    @staticmethod
    def score(t: Task, submission: str) -> float | None:
        hf_repo_id = t["model"]["hf_repo_id"]
        variant = t["variant"]
        name = t["model"]["name"]

        ssh_client = aux_vm_access.ssh_client()
        ssh_client.open_sftp().put("./score.py", "resources/score.py")

        try:
            (stdin, stdout, stderr) = ssh_client.exec_command(
                f". myenv/bin/activate && \
                    python3 resources/score.py --hf_repo_id {hf_repo_id} \
                    --name {name} --variant {variant}")
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                lines = stdout.readlines()
                score = float(lines[-1].strip())
                assert 0 <= score <= 1
            else:
                logging.info(f"Error running score script on aux vm. \
                    Error: {stderr.readlines()}")
                score = 0
        except:
            logging.info("Error running score script on aux vm")
            score = 0

        return score

    @staticmethod
    def teardown(t: Task) -> None:
        pass
