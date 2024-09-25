# Instructions to run the `langchain-react-baseline-with-write-file` on the Prompt Injection Task

## Create the task environment 

1. Run `cd workbench/`
2. Run `npm run task -- ../prompt_injection_uplift/ <TASK_NAME>` where <TASK_NAME> is one of `gpt3_easy`, `gpt3_medium`, `gpt3_hard`, and `gpt3_very_hard`. These tasks are defined in `prompt_injection_uplift/prompt_injection_uplift.py`
3. Copy the container id from the output of the above comand to pass to the agent.

## Run the langchain ReAct agent on the created task

1. Run `npm run agent -- <CONTAINER_NAME> example-agents/langchain-react-baseline-with-write-file/ "pip install -q -r requirements.txt && env OPENAI_API_KEY='sk-proj-...'  MAX_STEPS=<MAX_STEPS> python -u main.py"`

Note that you also need to pass MAX_STEPS. Currently they are mapped inside `prompt_injection_uplift/prompt_injection_uplift.py` as follows:

| Variant | Num of Steps |
|---|---|
| Easy | 30 |
| Medium | 50 |
| Hard | 70 |
| Very Hard | 100 |

MAX_STEPS depends on which variant of the task you are running the agent on. It is recommended that the steps increase with increased complexity of the task