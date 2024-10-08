To run the ReAct agent on the MCQA Tasks:


First start the Task:
```
git clone https://github.com/samizdis/impact-academy
cd drivers && npm install
cd ..\workbench && npm install
npm run task --  ../mcqa <variant>
```

where `<variant>` is `mmlu` for the MMLU variant.


And then run the agent at the task:

`npm run agent -- <container name> example-agents/<agent> "pip install -q -r requirements.txt && env OPENAI_API_KEY='sk-proj-U6b....AA' python -u main.py" | tee logfile.log`

Where `<agent>` is `langchain-react-baseline` for the "ReAct" agent in the paper, and `langchain-react-baseline-with-ssh` for the "ReAct + SSH" agent.
