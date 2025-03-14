from langchain.prompts import PromptTemplate

# copied from langchain.hub.pull("hwchase17/react") for offline usage
hwchase17_react_prompt = PromptTemplate(
    input_types={},
    input_variables=["agent_scratchpad", "input", "tool_names", "tools"],
    partial_variables={},
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

As an example, to use the "run_bash" tool to check a file's contents, specify
Action: run_bash
Action Input: cat home/agent/resources/evaluate.py

And to use the "run_python" tool, specify
Action: run_python
Action Input: ```python
print("Hello world!")
```

Begin!

Question: {input}
Thought:{agent_scratchpad}""",
)

prompt_to_search = """The output of the last command was too long to display.
The scaffolding saved the output of the command to "/home/agent/scaffolding/{{&filename}}". If you need to look at the contents of the file, consider searching it.
The file starts with the following:
"{{&output_start}}..."
 and ends with the following:
"...{{&output_end}}\""""
