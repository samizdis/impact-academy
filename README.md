# impact-academy

## Setup
Make sure docker-engine is running. Then run:

```
git clone https://github.com/samizdis/impact-academy
cd drivers && npm install
cd ..\workbench && npm install
npm run task -- ..\basic_factoring_task main
```

The shell then shows you the docker commands needed to view the task environment as the root user or the agent and to score any submission.

## Run an agent on the task

```
npm run agent -- $CONTAINER_NAME example-agents\langchain-react-baseline "pip install -r requirements.txt && env OPENAI_API_KEY='sk-xxx' python main.py"
```