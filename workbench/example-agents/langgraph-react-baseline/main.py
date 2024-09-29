from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from commands import _run_bash, _run_python

model = ChatOpenAI(model="gpt-4o", temperature=0)
memory = MemorySaver()

TASK_INSTRUCTIONS_PATH = "/home/agent/instructions.txt"
TASK_SUBMISSION_PATH = "/home/agent/submission.txt"
LOGGING_PATH = "/home/agent/logs.log"

@tool
def run_python(code):
    """Run the code in an interactive python session on the server"""
    global total_interactive_python_calls
    code = code.strip()
    if code.startswith("```python") and code.endswith("```"):
        code = code[9:-3].strip()
    result = _run_python(code)
    total_interactive_python_calls += 1
    return result


@tool
def run_bash(command):
    """Run the shell command on the server"""
    global total_bash_commands
    command = command.strip()
    if command.startswith("`") and command.endswith("`"):
        command = command[1:-1].strip()
    result = _run_bash(command)
    total_bash_commands += 1
    return result

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def get_task():
    with open(TASK_INSTRUCTIONS_PATH, "r") as f:
        return f.read()

def main():
    tools = [run_bash, run_python]
    model.bind_tools(tools)
    graph = create_react_agent(model, tools=tools, checkpointer=memory)
    inputs = {"messages": [("user", get_task())]}
    stream = graph.invoke(inputs)
    print_stream(stream)

if __name__ == '__main__':
    main()