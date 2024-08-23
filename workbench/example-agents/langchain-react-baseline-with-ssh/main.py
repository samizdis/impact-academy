# Based on https://python.langchain.com/docs/modules/agents/agent_types/react
from typing import Any, List

from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentFinish
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain.tools import tool
from openai import OpenAI
from commands import _run_python, _run_bash, _run_bash_on_vm
from templates import hwchase17_react_prompt
from ssh_tool import ShellHandler

with open("/home/agent/.ssh/hostname", "r") as f:
    hostname = f.read().strip()

with open("/home/agent/.ssh/username", "r") as f:
    username = f.read().strip()

shell_handler = ShellHandler(
    hostname=hostname,
    username=username,
    key_filename="/home/agent/.ssh/vm",
)

TASK_INSTRUCTIONS_PATH = "/home/agent/instructions.txt"
TASK_SUBMISSION_PATH = "/home/agent/submission.txt"

openai_client = OpenAI()


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
        return (
            openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="gpt-4o",
                n=1,
                temperature=0.1,
                max_tokens=1024,
                stop=stop,
            )
            .choices[0]
            .message.content
        )


@tool
def run_python(code):
    """Run the code in an interactive python session on the server"""
    code = code.strip()
    if code.startswith("```python") and code.endswith("```"):
        code = code[9:-3].strip()
    result = _run_python(code)
    return result


@tool
def run_bash(command):
    """Run the shell command on the local server"""
    command = command.strip()
    if command.startswith("`") and command.endswith("`"):
        command = command[1:-1].strip()
    result = _run_bash(command)
    return result

@tool
def run_bash_on_vm(command):
    """Run the shell command on the remote VM server."""
    command = command.strip()
    if command.startswith("`") and command.endswith("`"):
        command = command[1:-1].strip()
    result = _run_bash_on_vm(shell_handler, command)
    return result


def get_task():
    with open(TASK_INSTRUCTIONS_PATH, "r") as f:
        return f.read()


def main(*args):
    print("STARTING AGENT")
    task_string = get_task()
    print("Task:", task_string)

    llm = CustomLLM()
    tools = [run_bash_on_vm, run_bash]
    prompt = hwchase17_react_prompt

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=30,
    )

    agent_executor.invoke({"input": task_string})




if __name__ == "__main__":
    main(main)