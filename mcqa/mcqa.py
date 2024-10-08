from pathlib import Path
import subprocess, sys, os, shutil

from variants import Task, task_variants
import logging

from metr_task_standard.types import VMSpec


logging.basicConfig(filename="/root/logfile.log",
                    filemode="a",
                    level=logging.DEBUG,  # Minimum level of messages to log
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
        "OPENAI_API_KEY"
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
        # Download MMLU to root directory, to copy into /home/agent during start()
        # run_command('wget -P /root https://people.eecs.berkeley.edu/~hendrycks/data.tar && tar xf /root/data.tar')
        # run_command('pip install -r requirements.txt')
        for v in task_variants.keys():
            requirements_file = Path(v) / 'requirements.txt'
            logging.warning(requirements_file.absolute())
            if requirements_file.exists():
                run_command(f'pip install -r {requirements_file}')

        run_command(f'pip install -r langchain-react-baseline/requirements.txt')

        shutil.copytree('langchain-react-baseline', '/home/agent/langchain-react-baseline')


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
        return t["instructions"]

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
        return None

    # `start` is called after creating a Docker container for a run, but before the agent process starts.
    @staticmethod
    def start(t: Task) -> None:   

        # add the LLM API key to the VM
        openai_key = os.environ['OPENAI_API_KEY']
        if openai_key is None or len(openai_key) == 0:
            logging.error("Environment variable OPENAI_API_KEY is not set.")
        else:
            with open("/home/agent/keys.cfg", "w") as f:
                f.write(f"OPENAI_API_KEY: '{openai_key}'\n")

        if t['variant'] == 'mmlu':
            import datasets
            import pandas as pd

            mmlu_test = pd.DataFrame(datasets.load_dataset('cais/mmlu', 'all')['test'])
            mmlu_test_without_answers= (mmlu_test
                                        .assign(**{f"choice_{chr(ord('a') + i)}": mmlu_test['choices'].apply(lambda x: x[i])
                                                   for i in range(4)})
                                        .drop(columns=['choices', 'answer'])
                                        )
            mmlu_test_without_answers.to_csv('/home/agent/mmlu_test_without_answers.csv', index=False)
            mmlu_test[['answer']].to_csv('/home/agent/mmlu_test_answers.csv')

            os.symlink('/home/agent/mmlu_test_without_answers.csv',
                       '/home/agent/test_questions.csv')
            os.symlink('/home/agent/mmlu_test_answers.csv',
                       '/home/agent/test_answers.csv')

    # The TaskFamily class MAY define a static method called `score`.
    # `score` is called after the agent submits a solution.
    # `score` should return a float between 0 and 1, or `None` to indicate that manual scoring is required.
    # If not defined, the score defaults to `None`.

    @staticmethod
    def score(t: Task, submission: str) -> float | None:
        """
        Score function returns a float between 0 and 1.

        Points are gained for:
            a) ....
                (0.xyz points per model)
            b) ....
                (0.xyz points per model)

        At each step, if the agent fails then scoring stops (i.e. the agent does not get points
        for step d) unless steps a) through c) were performed correctly.
        """

        return None

    # The TaskFamily class MAY define a static method called `teardown`.
    # Task Standard implementations MAY call teardown to clean up external resources created during task
    # environment setup.
    @staticmethod
    def teardown(t: Task) -> None:
        pass
