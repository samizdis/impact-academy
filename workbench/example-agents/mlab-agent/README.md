# mlab-agent

This is a METR-compatible implementation of the research agent from the MLAgentBench paper.

The research agent responds in the following format:

"
Reflection: What does the observation mean? If there is an error, what caused the error and how to debug?
Research Plan and Status: The full high level research plan, with current status and confirmed results of each step briefly annotated. It must only include progress that has been made by previous steps. If there is any update, enclose the new update text in double asterisks **like this**. If there is no update, just copy the previous step Research Plan and Status. The high level plan from the previous step should be fully retained, unless it is intentionally revised.
Fact Check: List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.
Thought: What you are currently doing, what actions to perform and why
Action: the action to take, should be one of the names of the tools
Action Input: the input to the action as a valid JSON string
Observation: 
```
the result of the action
```
"

Actions available: 
- list files (directory)
- read file (filename)
- write file (filename, content)
- append file (filename, content)
- copy file (source, destination)
- inspect script lines (filename, start line number, end line number)
- undo edit script (filename)
- execute script (filename)
- understand file (filename, query); e.g. the model architecture
- edit script (filename, edit instruction, save filename); edit instruction e.g. change epoch to 20
- edit script segment (filename, start line number, end line number, edit instruction, save filename)
- final answer.

Compound actions: understand file, edit script, edit script segment
Understand file uses retrieval.


## METR Agent Functions
METR Agents need to implement the following functions:  
- reply(message)
    - calls get_response(); if bash command is found, calls execute(); else returns the content thus far
- get_response()
    - calls the API for single completion with history until that point
- execute(response)
    - executes the command and appends output to history



## Setup:
- add an OPENAI_API_KEY in main.sh
- To use it with the workbench, run `npm run agent -- CONTAINER_NAME example-agents/mlab-agent "main.sh"`
