# Setup
1. Clone the repo:
```
git clone https://github.com/samizdis/impact-academy
```

2. Create a .env file inside the main directory with the following variables:
```
KAGGLE_USERNAME=<username>
KAGGLE_KEY=<key>
OPENAI_API_ORG=<org-...>
OPENAI_API_KEY=<sk-...>
```

# Running agents on MLAB Tasks
To run the ReAct agent on the MLAB tasks:

First start the Task:
```
cd drivers && npm install
cd ..\workbench && npm install
npm run task --  ../mlagentbench <variant>
```

The following variants are available:
- easy-logs-complete-1
- easy
- hard.

```npm run task``` will give you the `<container name>`.

To run an agent at the task:

`npm run agent -- <container name> example-agents/langchain-react-baseline "pip install -q -r requirements.txt && env OPENAI_API_KEY='sk-...' python -u main.py" | tee logfile.log`


