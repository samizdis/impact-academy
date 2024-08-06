# mlab-agent

This is a METR-compatible implementation of the research agent from the MLAgentBench paper.

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



To use it with the workbench, run `npm run agent -- CONTAINER_NAME example-agents/mlab-agent "pip install -r requirements.txt && env OPENAI_API_KEY='sk-...' python main.py"`
