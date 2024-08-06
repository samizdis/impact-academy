from typing import Any, List
from datetime import datetime
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain_core.agents import AgentFinish
# from langchain_core.callbacks import CallbackManagerForLLMRun
# from langchain_core.language_models import LLM
# from langchain.tools import tool
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", filename="agent_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")))

openai_client = OpenAI()

# TODO: A utility to easily add new tools
tools = [
    {
        "Action": "run_python",
        "Action Inputs": {
            "script_name": "[a valid python script name with relative path to current directory if needed]"
        },
        "Action Observation": "The observation will be the output of the script or errors. In case there is no observation from the execution, the environment will provide the message 'The script executed successfully but generated no output'",
    },
    {
        "Action": "run_bash",
        "Action Inputs": {
            "command": "[a valid bash command]"
        },
        "Action Observation": "The observation will be the output of the command or errors.  In case there is no observation from the execution, the environment will provide the message 'The script executed successfully but generated no output'",
    },
]

COST_PER_CONTEXT_TOKEN = {
    'gpt-3.5-turbo': 0.0001,
    'gpt-4o': 0.0001,
}

COST_PER_GENERATED_TOKEN = {
    'gpt-3.5-turbo': 0.0001,
    'gpt-4o': 0.0001,
}

def get_tools_string(tools):
    result = ""
    for tool in tools:
        result += f"Action: {tool['Action']}\nAction Inputs:\n"
        for input_key, input_value in tool['Action Inputs'].items():
            result += f"  {input_key}: {input_value}\n"
        result += f"Action Observation: {tool['Action Observation']}\n\n"
    return result.strip()


# task_description = "Answer the following questions as best you can. You have access to the following tools:"
with open("task_description.txt", "r") as f:
    task_description = f.read()

REACT_PROMPT = """
You are a helpful assistant. You have access to a set of tools. You can use the following tools by specifying the Action and Action Inputs (if any), the expected Observation is also given:

{tools}

Task: {task}

Use the following format:
Question: the question you must ask to solve the Task
Thought: what do you think you should do to answer the Question
Plan: a short outline of what you broadly plan to do to answer the Question
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: what did you observe after taking the action
... (this Thought/Plan/Action/Action Input/Observation can repeat N times)
Final Thought: I have successfully completed the task

""".format(tools=get_tools_string(tools), task=task_description, tool_names=", ".join([tool['Action'] for tool in tools]))


class TokenCount:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.context_token_count = 0
        self.generated_token_count = 0

    def __repr__(self) -> str:
        return f"TokenCount(model_name={self.model_name}, context_token_count={self.context_token_count}, generated_token_count={self.generated_token_count})"

class MetaCost:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.context_token_count = 0
        self.generated_token_count = 0
        self.cost = 0

    def __repr__(self) -> str:
        return f"MetaCost(model_name={self.model_name}, context_token_count={self.context_token_count}, generated_token_count={self.generated_token_count}, cost={self.cost})"

class Episode:
    def __init__(self) -> None:
        self.question = ""
        self.thought = ""
        self.plan = ""
        self.action = ""
        self.action_input = ""
        self.observation = ""
        self.finished_task = False

    def __repr__(self) -> str:
        return f"Episode(question={self.question}, thought={self.thought}, plan={self.plan}, action={self.action}, action_input={self.action_input}, observation={self.observation}, finished_task={self.finished_task})"    
    
    def __str__(self) -> str:
        return f"Episode(question={self.question}, thought={self.thought}, plan={self.plan}, action={self.action}, action_input={self.action_input}, observation={self.observation}, finished_task={self.finished_task})"

class SystemPrompt:
    def __init__(self, content) -> None:
        self.content = content

class Environment:
    def __init__(
        self, max_steps=20, max_cost=200, agent_llm="gpt-4o", agent_llm_temp=0.1
    ) -> None:
        logging.info("Creating environment")
        self.max_steps = max_steps
        self.max_cost = max_cost

        self.agent_llm = agent_llm
        self.agent_llm_temp = agent_llm_temp

        self.agent = None

        self.agent_model_token_count = TokenCount(self.agent_llm)

        system_prompt = SystemPrompt(REACT_PROMPT)
        self.history: List[Episode] = [{
            'role': 'system',
            'content': system_prompt.content
        }]

        logging.info("Environment created")
        logging.info("Max steps: %s", self.max_steps)
        logging.info("Max cost: %s", self.max_cost)
        logging.info("Agent LLM: %s", self.agent_llm)
        logging.info("Agent LLM Temp: %s", self.agent_llm_temp)
        logging.info("System Prompt: %s", REACT_PROMPT)

        self.task_finished = False


    def create_react_agent(self):
        self.agent = Agent(self.agent_llm, self.agent_llm_temp)

    def run_react_agent(self):
        current_step = 0
        current_cost = 0
        while (
            (not self.task_finished)
            and (current_step < self.max_steps)
            and (current_cost < self.max_cost)
        ):
            episode, meta = self.agent.run(self.history)
            # summarised_episode = self.summarise_episode(episode)
            # self.history.append(summarised_episode)
            self.history.append(str(episode))

            current_cost += meta.cost
            self.agent_model_token_count.context_token_count += meta.context_token_count
            self.agent_model_token_count.generated_token_count += meta.generated_token_count

            current_step += 1

            if episode.finished_task:
                # TODO: logging
                self.task_finished = True

        if current_step >= self.max_steps:
            print("Exited due to max steps reached")
        
        if current_cost >= self.max_cost:
            print("Exited due to max cost reached")
        
        if self.task_finished:
            print("Task completed successfully")

class Agent:
    def __init__(self, llm_name, model_temp) -> None:
        self.llm_name = llm_name
        self.model_temp = model_temp

    def run(self, history):
        """
        Input: list of all past episodes
        Output: The next episode and the meta objection containing token usage
        """
        meta = MetaCost(self.llm_name)
        episode = Episode()
        # call the openai model with chat completion and get the response
        response = openai_client.chat.completions.create(
            model=self.llm_name,
            messages=history,
            temperature=self.model_temp
        )
        meta.context_token_count += response.usage.prompt_tokens
        meta.generated_token_count += response.usage.completion_tokens
        # parse the response, the fields will be as per fields in REACT_PROMPT
        data = response.choices[0].message.content
        logging.info("Response from LLM: %s", data)
        logging.info("Token usage: %s", meta)

        # extract and trim the substring from Question: to Thought:
        question = data[data.find("Question:")+len("Question:"):data.find("Thought:")]
        episode.question = question.strip()
        thought = data[data.find("Thought:")+len("Thought:"):data.find("Plan:")]
        episode.thought = thought.strip()
        plan = data[data.find("Plan:")+len("Plan:"):data.find("Action:")]
        episode.plan = plan.strip()
        action = data[data.find("Action:")+len("Action:"):data.find("Action Input:")]
        episode.action = action.strip()
        action_input = data[data.find("Action Input:")+len("Action Input:"):data.find("Observation:")]
        episode.action_input = action_input.strip()
        observation = data[data.find("Observation:")+len("Observation:"):data.find("Final Thought:")]
        episode.observation = observation.strip()
        # final_thought = data[data.find("Final Thought:")+len("Final Thought:"):]
        # episode.finished_task = True

        # execute the Action if any
        


        if "Final Thought:" in data:
            episode.finished_task = True
        
        context_cost = meta.context_token_count * COST_PER_CONTEXT_TOKEN[self.llm_name]
        generated_cost = meta.generated_token_count * COST_PER_GENERATED_TOKEN[self.llm_name]
        meta.cost = context_cost + generated_cost

        return episode, meta

if __name__ == "__main__":
    environment = Environment(agent_llm="gpt-3.5-turbo")
    environment.create_react_agent()
    environment.run_react_agent()
