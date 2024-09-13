# Based on https://python.langchain.com/docs/modules/agents/agent_types/react
from typing import Any, List

from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentFinish
from langchain_core.callbacks import CallbackManagerForLLMRun, FileCallbackHandler
from langchain_core.language_models import LLM
from langchain.tools import tool
from openai import OpenAI
import sys
from ssh_tool import ShellHandler
import argparse

shell_handler = None

def setup_ssh_tool():
    global shell_handler
    with open("/home/agent/.ssh/hostname", "r") as f:
        hostname = f.read().strip()

    with open("/home/agent/.ssh/username", "r") as f:
        username = f.read().strip()

    shell_handler = ShellHandler(
        hostname=hostname,
        username=username,
        key_filename="/home/agent/.ssh/vm",
    )

from commands import _run_python, _run_bash, _run_bash_on_vm
from templates import hwchase17_react_prompt

TASK_INSTRUCTIONS_PATH = "/home/agent/instructions.txt"
TASK_SUBMISSION_PATH = "/home/agent/submission.txt"
LOGGING_PATH = "/home/agent/logs.log"

openai_client = OpenAI()


# Token and cost tracking variables
total_input_tokens = 0
total_output_tokens = 0
total_cost = 0
total_bash_commands = 0
total_interactive_python_calls = 0
total_api_calls = 0

# Function to estimate cost
def estimate_cost(input_tokens, output_tokens):
    input_cost = (5 * input_tokens) / 1_000_000  # $5 per 1M tokens for input
    output_cost = (15 * output_tokens) / 1_000_000  # $15 per 1M tokens for output
    return input_cost + output_cost


class CustomLLM(LLM):
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        global total_input_tokens, total_output_tokens, total_cost, total_api_calls
        response = openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="gpt-4o",
                n=1,
                temperature=0,
                max_tokens=1024,
                stop=stop,
            )
        # Update token count, api call count, cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        total_api_calls += 1
        
        total_cost += estimate_cost(input_tokens, output_tokens)
        return (response.choices[0].message.content)


class AgentCallbackHandler(BaseCallbackHandler):
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        with open(TASK_SUBMISSION_PATH, "w") as f:
            f.write(finish.return_values["output"])


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
    """Run the shell command locally.
    """
    global total_bash_commands
    command = command.strip()
    if command.startswith("`") and command.endswith("`"):
        command = command[1:-1].strip()
    result = _run_bash(command)
    total_bash_commands += 1
    return result

@tool
def run_bash_on_vm(command):
    """Run the shell command on the remote VM server.
    This will run a command on the hostname at ~/.ssh/hostname, as ~/.ssh/username .
    These are the same values as in ~/ssh_args .
    Do not use the command in ~/ssh_args to run commands on the remote VM server,
    instead use this command.
    You can still use the information in ~/ssh_args to move files between this machine
    and the remote VM server.

    This command will keep track of environment variables
    (like Python virtual environments and current working directory).
    If you change working directory using this command,
    the next execution of run_bash_on_vm will also start from the new working directory.
    """
    global shell_handler
    command = command.strip()
    if command.startswith("`") and command.endswith("`"):
        command = command[1:-1].strip()
    result = _run_bash_on_vm(shell_handler, command)
    return result


@tool
def write_file(data) -> str:
    """Input:
    data: A valid JSON string containing `file_name` and `content` fields. 
    `content` field must contain the text to be written to the file. 
    Use fully qualified name for `file_name`. None of `file_name` or `content` can be empty. 
    No other control character or invalid JSON character is allowed in data.
    """
    import json, re

    # pre-process the data string to remove any redundant characters
    if data.startswith("```") and data.endswith("```"):
        data = data[3:-3].strip()
    if data.startswith("json"):
        data = data[4:].strip()
    sanitised_data = re.sub(
        r"[\x00-\x1f\x7f]", "", data
    )  # sanitise the data to remove control characters

    try:
        data_json = json.loads(sanitised_data)
    except json.JSONDecodeError:
        return "Unable to parse given inputs into JSON with fields file_name and content. You must provide valid JSON string with only two fields file_name and content and no other control characters."
    except Exception as e:
        # raise e
        return "Error while trying to parse the given inputs into JSON with fields file_name and content. You must provide valid JSON string with only two fields file_name and content and no other control characters."

    file_name = data_json["file_name"]
    content = data_json["content"]
    # pre-process the content string to remove any redundant characters
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()
    if content.startswith("python"):
        content = content[6:].strip()
    try:
        with open(file_name, "w") as f:
            f.write(content)
        observation = f"File {file_name} written successfully."
        return observation
    except Exception as e:
        return "Error while trying to write to file {}".format(file_name)

def get_task():
    with open(TASK_INSTRUCTIONS_PATH, "r") as f:
        return f.read()


def main(provide_ssh_tool=False):
    print("STARTING AGENT", file=sys.stderr)
    task_string = get_task()
    print("Task:", task_string)
    
    if provide_ssh_tool:
        setup_ssh_tool()
        tools = [run_python, run_bash, run_bash_on_vm, write_file]
    else:
        tools = [run_python, run_bash, write_file]

    llm = CustomLLM()
    prompt = hwchase17_react_prompt

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[AgentCallbackHandler(), FileCallbackHandler(LOGGING_PATH)],
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=30,
    )

    agent_executor.invoke({"input": task_string})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provide_ssh_tool", action='store_true')
    args = parser.parse_args()
    import time
    start = time.time()
    main(args.provide_ssh_tool)
    end = time.time()
    print(f"Time taken in minutes: {(end - start) / 60}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total tokens: {total_input_tokens + total_output_tokens}")
    print(f"Total cost (estimated): ${total_cost:.4f}")
    print(f"Total bash commands executed: {total_bash_commands}")
    print(f"Total interactive python calls: {total_interactive_python_calls}")
    print(f"Total API calls: {total_api_calls}")
