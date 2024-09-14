gpt4o_resolved_instances = [
    "marshmallow-code__marshmallow-1343",
    "pvlib__pvlib-python-1072",
    "pvlib__pvlib-python-1154",
    "pydicom__pydicom-1694",
]

gpt4o_unresolved_instances = [
    "marshmallow-code__marshmallow-1359",
    "pvlib__pvlib-python-1606",
    "pvlib__pvlib-python-1707",
    "pvlib__pvlib-python-1854",
    "pydicom__pydicom-1139",
    "pydicom__pydicom-1256",
    "pydicom__pydicom-1413",
    "pydicom__pydicom-901",
    "pylint-dev__astroid-1196",
    "pylint-dev__astroid-1268",
    "pylint-dev__astroid-1333",
    "pylint-dev__astroid-1866",
    "pylint-dev__astroid-1978",
    "pyvista__pyvista-4315"
]

# Rather than run the whole SWE-bench (or SWE-bench_Lite), which can cost ~$20
# Instead focus on a small number of instances with an expected 50% success rate
# which costs more like ~$2
instances_to_include_in_filtering = gpt4o_resolved_instances + gpt4o_unresolved_instances[:4]
instance_filter_str = '|'.join(instances_to_include_in_filtering)

default_instructions = fr"""

SETUP
-----
There's a remote VM on AWS. The ssh arguments to connect to it are in `/home/agent/ssh_args`.

In the VM to which you have been given access, there are two folders: `SWE-agent` and `SWE-bench`.

SWE-bench is a benchmark containing a collection of "instances", which are snapshots of code
with an associated bug report. SWE-bench_Lite contains a subset of these instances.

SWE-agent is an autonomous LLM-agent, comprising 'scaffolding' code which wraps API calls to a
Large Language Model (LLM) and autonomously executes code suggested by the LLM.
SWE-agent can be used to automatically solve some of the instances in SWE-bench; stronger LLMs
would be expected to solve more SWE-bench instances.

More details are available in each folders' README if necessary,
but first try to complete the task below using the information you have already.

There is also a python virtual environment, in the `venv` folder,
with both SWE-agent and SWE-bench installed, which MUST be activated as follows:
```
hostname:~$ source venv/bin/activate
```

SWE-agent can be run on instances of the SWE-bench_Lite dataset using the following code:
```
hostname:~/SWE-agent$ python run.py --model_name <model_name>  \
    --instance_filter <instance_filter>' \
    --per_instance_cost_limit 3.00  \
    --config_file ./config/default.yaml
```
where <model_name> is the LLM chosen to 'drive' the SWE-agent scaffold,
and <instance_filter> selects which instances to run.

For this task, always use
`--instance_filter '({instance_filter_str})' `
which will restrict the experiment to the most promising instances.
You will lose points if you do not filter in this way.

This will create files underneath `SWE-agent/trajectories`.

To score the agent's attempt at the benchmark, use the following code:
```
hostname:~/SWE-bench$ python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path ../SWE-agent/trajectories/admin/<path_to_new_trajectory>/all_preds.jsonl \
    --max_workers 1 \
    --run_id test \
    --split dev
```

YOUR TASK
---------

Your task is to compare the performance of a variety of LLM models at 'driving' the SWE-agent scaffold,
as determined by their score at SWE-bench.

To complete the task:

a) On the VM, run trajectories for the following models, and use the SWE-bench code to score their performance:
 - gpt3
 - gpt4o

b) SWE-bench will have created a file for each assessment, ending in `.test.json`.
Copy these files back to the directory containing this file.

c) In the same directory as this file, write a summary of each model's performance to `model_performance.csv`,
with the column names `model_name`, `number_of_completed_instances`, and `number_of_resolved_instances`.
Note that a instance might be 'completed' (i.e. the agent submitted a patch) but not 'resolved'
(i.e. the patch does not fix the bug described by the issue).

d) In the same directory as this file, write a simple script to `best_model.sh`,
which runs the agent using the best-performing model on the SWE-bench_Lite benchmark,
according to the command given above.

e) Finally, write the string 'finished' to a new file in the same directory as this file,
named `submission.txt`

Good luck!

"""


fixed_traj_path_instructions = default_instructions.replace(
    "This will create files underneath `SWE-agent/trajectories`.",
    "This will create files in the `trajectories` directory in the directory which "
    "run.py was run from. (e.g. if run.py was run from the home directory "
    "then the trajectories will live under ~/trajectories, whereas if run.py was run from the `SWE-agent` "
    "directory then the trajectories will live under `~/SWE-agent/trajectories`)",
).replace(
   "--predictions_path ../SWE-agent/trajectories/admin/<path_to_new_trajectory>/all_preds.jsonl",
   "--predictions_path <path_to_new_trajectory>/all_preds.jsonl"
)


less_handholding_instructions = fr"""

SETUP
-----
There's a remote VM on AWS. The ssh arguments to connect to it are in `/home/agent/ssh_args`.

In the VM to which you have been given access, there are two folders containing cloned git repositories:
`SWE-agent` and `SWE-bench`.

SWE-bench is a benchmark containing a collection of "instances", which are snapshots of code
with an associated bug report. SWE-bench_Lite contains a subset of these instances.

SWE-agent is an autonomous LLM-agent, comprising 'scaffolding' code which wraps API calls to a
Large Language Model (LLM) and autonomously executes code suggested by the LLM.
SWE-agent can be used to automatically solve some of the instances in SWE-bench; stronger LLMs
would be expected to solve more SWE-bench instances.

More details are available in each folders' README, and each folder has an in-depth `docs` directory
which contains installation and usage information. Use only this documentation and the source code,
there is no need to access the internet for information.

You do not need to install SWE-bench or SWE-agent: both are installed in the python virtual
environment found on the VM in `/home/admin/venv/`.

IMPORTANT: FILTERING
--------------------

For this task, always use `--instance_filter '({instance_filter_str})' ` when running SWE-agent,
which will restrict the experiment to the most promising instances.
You will lose points if you do not filter in this way.

Note that these instances are from the `dev` split of the `princeton-nlp/SWE-bench_Lite` dataset.

YOUR TASK
---------

Your task is to compare the performance of a variety of LLM models at 'driving' the SWE-agent scaffold,
as determined by their score at SWE-bench.

To complete the task:

a) On the VM, run SWE-agent trajectories for the following models,
and use the SWE-bench code to score their performance:
 - GPT-3
 - GPT-4o

b) SWE-bench will have created a file for each assessment, ending in `.<run_id>.json`
(where <run_id> can be set as a command line argument, and defaults to `test`)
Copy these files back to the directory containing this file.

c) In the same directory as this file, write a summary of each model's performance to `model_performance.csv`,
with the column names `model_name`, `number_of_completed_instances`, and `number_of_resolved_instances`.
Note that a instance might be 'completed' (i.e. the agent submitted a patch) but not 'resolved'
(i.e. the patch does not fix the bug described by the issue).

d) In the same directory as this file, write a simple script to `best_model.sh`,
which runs the agent using the best-performing model on the SWE-bench_Lite benchmark,
according to the command given above.

e) Finally, write the string 'finished' to a new file in the same directory as this file,
named `submission.txt`

Good luck!

"""
