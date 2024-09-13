To run the ReAct agent, supply the max_steps as an environment variable in addition to the OpenAI API key

yarn run agent -- <CONTAINER_NAME> example-agents/langchain-react-baseline-with-write-file/ "pip install -q -r requirements.txt && env OPENAI_API_KEY='sk-proj-...'  MAX_STEPS=30 python -u main.py" | tee agent_run_log.txt

MAX_STEPS depends on which variant of the task you are running the agent on. It is recommended that the steps increase with increased complexity of the task