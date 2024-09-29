from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

_ = load_dotenv()

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

class AgentState(TypedDict):
  messages: Annotated[list[AnyMessage], operator.add]

class Agent:
  def __init__(self, model, tools, checkpointer, system_prompt=""):
    self.system_prompt = system_prompt
    graph = StateGraph(AgentState)
    graph.add_node("llm", self.call_openai)
    graph.add_node("action", self.take_action)
    graph.add_conditional_edges(
        "llm",
        self.exists_action,
        {True:"action", False:END}
    )
    graph.add_edge("action", "llm")
    graph.set_entry_point("llm")
    self.graph = graph.compile(checkpointer=checkpointer)
    self.tools = {t.name:t for t in tools}
    self.model = model.bind_tools(tools)
  
  def exists_action(self, state:AgentState):
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
      results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}

prompt = """You are a smart research assistant. Use the wikipedia to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model = ChatOpenAI(model="gpt-3.5-turbo")  #reduce inference cost
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
abot = Agent(model, [tool], memory, system_prompt=prompt)

messages = [HumanMessage(content="Can you give some data for the reasons you mentioned?")]
config = {"configurable": {"thread_id":"2"}}
print(abot.graph.get_state(config))
result = abot.graph.invoke({"messages": messages}, config)
print(abot.graph.get_state(config))

print(result['messages'][-1].content)