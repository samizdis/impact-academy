# langchain-react-baseline-v2

This agent is based on langchain-react-baseline, but has some additional instructions on how to deal with long outputs better.

This is a simple agent based on [the standard langchain implementation of the ReAct agent](https://python.langchain.com/docs/modules/agents/agent_types/react)

As written, this agent uses the `gpt-4-32k-0613` model.

To use it with the workbench, pass `"pip install -r requirements.txt && env OPENAI_API_KEY='sk-...' python main.py"` as a command to `npm run agent`.

To use it with the workbench, and with the ssh_tool, pass `"pip install -r requirements.txt && env OPENAI_API_KEY='sk-...' python main.py --provide_ssh_tool"` as a command to `npm run agent`.
