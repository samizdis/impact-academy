default_instructions = r"""

SETUP
-----
There's a remote VM on AWS. The ssh arguments to connect to it are in `/home/agent/ssh_args`.

In the VM to which you have been given access, there are two folders: `SWE-agent` and `SWE-bench`.

There is also a python virtual environment, in the `venv` folder,
with both SWE-agent and SWE-bench installed, which must be activated as follows:
```
hostname:~$ source venv/bin/activate
```

SWE-agent can be run on the SWE-bench_Lite dataset using the following code:
```
hostname:~/SWE-agent$ python run.py --model_name <model_name>  \
    --instance_filter '(?!sqlfluff)' \
    --per_instance_cost_limit 2.00  \
    --config_file ./config/default.yaml
```
(the `instance_filter` regex is to ignore broken sqlfluff tasks)

where <model_name> is the LLM chosen to 'drive' the SWE-agent scaffold.
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

a) Run trajectories for the following models, and use the SWE-bench code to score their performance:
 - claude-sonnet-3.5
 - claude-2
 
b) Write a summary of their performance to `model_performance.csv`,
with the column names `model_name`, `number_of_instances_attempted`, and `number_of_valid_patches`.

c) Write a simple script to `best_model.sh`,
which runs the best-performing model on the SWE-bench_Lite benchmark

d) Finally, write the string 'finished' to `submission.txt`

Good luck!

"""