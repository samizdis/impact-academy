from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

_ = load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system_prompt=""):
        self.system_prompt = system_prompt
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {
                True: "action",
                False: END
            }
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        last_message = state["messages"][-1]
        return len(last_message.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        next_message = self.model.invoke(messages)
        return {"messages": [next_message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t["name"] not in self.tools:
                result = "bad tool name, retry"
            else:
                result = self.tools[t["name"]].invoke(t['args'])
        results.append(
            ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            )
        )
        return {'messages': results}


system_prompt = """
You are a general AI assistant. \
I will ask you a question. \
Report your thoughts, and finish your answer \
with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. \
YOUR FINAL ANSWER should be a number OR as few words as possible \
OR a comma separated list of numbers and/or strings. \
If you are asked for a number, don't use comma to write your number \
neither use units such as $ or percent sign unles specified otherwise. \
If you are asked for a string, don't use articles, \
neither abbreviations (e.g. for cities), \
and write the digits in plain text unless specified otherwise. \
If you are asked for a comma separated list, apply the above rules depending \
of whether the element to be put in the list is a number or a string.
"""

model = ChatOpenAI(model="gpt-4o-mini")
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. \
        Input should be a valid python command. \
        If you want to see the output of a value, \
        you should print it out with `print(...)`.",
    func=python_repl.run,
)

agent = Agent(model, [repl_tool], system_prompt=system_prompt)
