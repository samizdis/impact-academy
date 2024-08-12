import subprocess
import os
import sys
import argparse
import logging

from dotenv import load_dotenv
import openai
import anthropic
from agent import Agent
from utils import Action, complete_text_openai
from environment import Environment

load_dotenv()

logging.basicConfig(level=logging.INFO)
MODEL = "gpt-3.5-turbo" # TODO: change this later to a more capable LLM

initial_prompt = """You are a helpful research assistant. You have access to the following tools:
{tools_prompt}

Research Problem: {task_description}

You do not know anything about this problem so far. 

Follow these instructions and do not forget them:
- First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
- Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. 
- Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation. 
- Performance numbers and estimates can only be confirmed and included in the status by running the code and observing the output.
- You should come up with a good experiment design that addresses the problem, and whenever applicable, define and measure the baseline performance of the relevant system or model before attempting any improvements.
- Follow the plan and try to achieve the goal as straightforwardly as possible.
- Highlight the supporting experiment results and reasoning before drawing any conclusions. 
- Do not try installing any new packages or libraries.
- If you believe you have solved the problem, you can use the Final Answer action to submit your answer. You can only submit once, so double check that you have achieved the goal before submitting.

Always respond in this format exactly:
{format_prompt}
Observation: 
```
the result of the action
```

"""

format_prompt_dict = {
    "Reflection": "What does the observation mean? If there is an error, what caused the error and how to debug?",
    "Research Plan and Status": "The full high level research plan, with current status and confirmed results of each step briefly annotated. It must only include progress that has been made by previous steps. If there is any update, enclose the new update text in double asterisks **like this**. If there is no update, just copy the previous step Research Plan and Status. The high level plan from the previous step should be fully retained, unless it is intentionally revised.",
    "Fact Check": "List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.",
    "Thought": "What you are currently doing, what actions to perform and why",
    "Action": "the action to take, should be one of the names of the tools",
    "Action Input": "the input to the action as a valid JSON string",
}


class ResearchAgent(Agent):
    """A not-so-simple agent based on the research agent introduced in MLAgentBench."""

    def __init__(self, args, env):
        print("STARTING RESEARCH AGENT", file=sys.stderr)
        super().__init__(args, env)
        self.valid_format_entries = ["Reflection", "Research Plan and Status", "Fact Check", "Thought", "Action", "Action Input"]
        '''if args.valid_format_entries:
            self.valid_format_entries = args.valid_format_entries'''
        
        #assert args.instructions, "Instructions not found. Initialize agent with instructions arg."

        self.initial_prompt = initial_prompt.format(tools_prompt=self.tools_prompt, 
                                                    tool_names=self.prompt_tool_names,  
                                                    task_description=env.research_problem, 
                                                    format_prompt="\n".join([f"{k}: {format_prompt_dict[k]}" for k in self.valid_format_entries]))
        self.history = [{"role": "system", "content": self.initial_prompt}]


    def run(self, env):
        last_steps = self.args.max_steps_in_context
        last_observation_step = self.args.max_observation_steps_in_context

        with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
            f.write(self.initial_prompt + "\n")

        while not env.is_final() and len(self.history_steps) < self.args.agent_max_steps:

            curr_step = len(self.history_steps)

            #### call LLM for next action ###

            ###########################################################
            #     construct prompt for LLM based on previous steps    #
            ###########################################################

            prompt = self.initial_prompt
            if curr_step > last_steps:
                if self.args.retrieval:  # note: the research agent has retrieval enabled

                    # retrieval action
                    relevant_history = env.execute(Action("Retrieval from Research Log", {"current_plan": ""}))

                    prompt += f"""
                                Here is a summary of relevant actions and observations you have done:
                                ```
                                {relevant_history}
                                ```
                                Here are the exact several steps you have done most recently (up to 3 steps):
                                """
            else:
                prompt += "\nNow let's start!\n\n"

            for idx in range(max(curr_step - last_steps, 0), curr_step):
                action_string = ""
                action_string = self.print_action(self.history_steps[idx]["action"], self.valid_format_entries)

                prompt += anthropic.AI_PROMPT + "\n"+ action_string + "\nObservation:"
                if curr_step - idx > last_observation_step:
                    prompt += "<Done>\n\n"
                else:
                    try:
                        prompt += "\n```\n" + self.history_steps[idx]["observation"] + "\n```\n\n"
                    except:
                        import pdb; pdb.set_trace()
                

            ###############################################
            #     call LLM until the response is valid    #
            ###############################################

            entries = None
            valid_response = False
            for _ in range(self.args.max_retries):
                log_file = os.path.join(self.log_dir , f"step_{curr_step}_log.log")
                completion = complete_text_openai(prompt, log_file, self.args.llm_name)

                try:
                    entries = self.parse_entries(completion, self.valid_format_entries)
                    assert entries["Action"].strip() in self.all_tool_names
                    valid_response = True
                except:
                    print("Step", curr_step, file=sys.stderr)
                    print("+++++++++++++++COMPLETION++++++++++++++++++++", completion, file=sys.stderr)
                    print(anthropic.AI_PROMPT + "\n" + completion + "\nObservation:\n", file=sys.stderr)
                    print("Response is invalid and discarded", file=sys.stderr)
                    prompt += "\n\n Your response was in incorrect format. Please provide a valid response with all entries: " + ", ".join(self.valid_format_entries) + "\n\n"
                else:
                    break
            if not valid_response:
                return "No valid response after max_retries"

            ########################################################
            #     postprocess LLM output and parse to env actions  #
            ########################################################
            rg = entries["Research Plan and Status"]
            action = entries["Action"].strip()
            raw_action_input = entries["Action Input"]

            new_research_plan_content = rg.strip("```") + "\n\n" 
            entries["Research Plan and Status"] = new_research_plan_content
            entries["Research Plan and Status"]= new_research_plan_content.replace("**", "")

            
            # parse the action input if we can ; other wise just return the original input and wait env to throw back an error
            parsing_error = ""
            try:
                action_input = self.parse_action_input(raw_action_input, self.action_infos[action])
            except Exception as e:
                action_input = raw_action_input
                parsing_error = str(e)
                

            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write("Step " + str(curr_step) + ":\n")
                f.write(anthropic.AI_PROMPT + "\n" + self.print_action(entries, self.valid_format_entries) + "\nObservation:\n")


            ########################################
            #         execute action in env        #
            ########################################

            if type(action_input) == dict:
                observation = env.execute(Action(action, action_input))
            else:
                # parsing failed, give agent parsing error
                usage = ",\n            ".join([f"{k}: [{v}]" for k, v in self.action_infos[action].usage.items()])
                usage = f"""{{
                        {usage}
                        }}"""
                invalid_action_error = f"The action input for {action} needs to be a valid json with proper entries. You may have missed the comma between entries or used triple quotes (json does not recognizes triple quotes). Please use the correct format and try again:\n{usage}"

                observation = "ActionInputParsingError: "+ parsing_error + "\n" + invalid_action_error


            #######################################################
            #               update history_steps                  #
            #######################################################

            # if observation is too long, we need to summarize it
            if len(observation) > 5000:
                log_file = os.path.join(self.log_dir , f"step_{curr_step}_summarize_observation_log.log")

                print("Observation is too long. Summarizing...", file=sys.stderr)
                observation = self.summarize_observation(self.print_action(entries, self.valid_format_entries), observation, log_file)

            self.history_steps.append({"step_idx": len(env.trace.steps), "action": entries, "observation": observation})

            ## filter out ActionInputParsingError if last step is not action input parsing error
            if not observation.startswith("ActionInputParsingError"):
                self.history_steps = [step for step in self.history_steps if not step["observation"].startswith("ActionInputParsingError")]

            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write("\n```\n" + self.history_steps[-1]["observation"] + "\n```\n\n")


            #######################################################
            #      write to research log for retrieval            #
            #######################################################
            if self.args.retrieval:
                summary_of_last_step = "Too long to summarize."
                for _ in range(self.args.max_retries):
                    try:
                        log_file = os.path.join(self.log_dir , f"step_{curr_step}_summary_log.log")
                        summary_of_last_step = self.summarize_action_and_observation(self.print_action(self.history_steps[-1]["action"], self.valid_format_entries), self.history_steps[-1]["observation"], log_file = log_file)
                        break
                    except Exception as e:
                        print(e)
                        print("Trying again.")

                action = "Append Summary to Research Log"
                action_input = { "content": "\n\nStep " + str(curr_step) + ":\n" + summary_of_last_step + "\n"}
                env.execute(Action(action, action_input))

            step_idx = len(env.trace.steps) - 1
            self.save(os.path.join(self.log_dir , f"agent_{step_idx}_{curr_step}.json"))

        if env.is_final():
            return "Finished due to env.is_final() == True"
        else:
            return "Finished due to agent max steps reached"


    ################### Helper functions #####################

    def summarize_observation(self, action, observation, log_file, bs = 10000):
        """ Summarize the observation if it is too long with a sliding window of size bs """

        bs = 10000
        blocks = [observation[i:i+bs] for i in range(0, len(observation), bs)]
        descriptions = []
        for idx, b in enumerate(blocks):
            start_line_number = bs*idx+1
            end_line_number = bs*idx+1 + len(b)
            prompt = f"""
                    {action}

                    The full observation is too long. Given this (partial) observation from character {start_line_number} to character {end_line_number}: 
                    ``` 
                    {b}
                    ```
                    Summarize the observation concisely in this format:
                    [Observation]: Summarize all relevant details in the observation objectively

                    Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
                    """

            completion = complete_text_openai(prompt, log_file=log_file +f"_{idx}")
            descriptions.append(completion)
        if len(descriptions) == 1:
            completion = descriptions[0]
        else:
            descriptions = "\n\n".join(["Segment {idx}: \n\n" + s for s in descriptions])
            prompt = f"""
                    {action}

                    The full observation is too long. 
                    Given summaries for each segments of the whole observation, summarize to get a cohesive description of the entire observation.
                    {descriptions}

                    Summarize the observation concisely in this format:
                    [Observation]: Summarize all relevant details in the observation objectively

                    Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
                    """

            completion = complete_text_openai(prompt, log_file=log_file)
        try:
            return completion.split("[Observation]:")[1]
        except:
            return completion

    @staticmethod
    def summarize_action_and_observation(action, observation, **kwargs):
        """ Summarize the action and observation to an entry in the research log """

        prompt = f"""Given your action and the observation: 
        {action} 
        [Observation]:
        ```
        {observation}
        ```
        Summarize your action and the observation in this format:
        [Reasoning]: Summarize the reasoning behind the action
        [Action]: Summarize all relevant details of the action objectively
        [Observation]: Summarize all relevant details in the observation objectively
        Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
        """

        summary = "[Reasoning]:" + complete_text_openai(prompt, log_file=kwargs["log_file"]).split("[Reasoning]:")[1]
        return summary



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="debug", help="task name")
    parser.add_argument("--log-dir", type=str, default="/home/agent/logs", help="log dir")
    parser.add_argument("--work-dir", type=str, default="/home/agent", help="work dir")
    parser.add_argument("--max-steps", type=int, default=50, help="number of steps")
    parser.add_argument("--max-time", type=int, default=5* 60 * 60, help="max time")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--python", type=str, default="python", help="python command")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument("--resume-step", type=int, default=0, help="the step to resume from")

    # general agent configs
    parser.add_argument("--agent-type", type=str, default="ResearchAgent", help="agent type")
    parser.add_argument("--llm-name", type=str, default="gpt-4o", help="llm name")
    parser.add_argument("--fast-llm-name", type=str, default="gpt-4o", help="llm name")
    parser.add_argument("--edit-script-llm-name", type=str, default="gpt-4o", help="llm name")
    parser.add_argument("--edit-script-llm-max-tokens", type=int, default=4000, help="llm max tokens")
    parser.add_argument("--agent-max-steps", type=int, default=50, help="max iterations for agent")

    # research agent configs
    parser.add_argument("--actions-remove-from-prompt", type=str, nargs='+', default=[], help="actions to remove in addition to the default ones: Read File, Write File, Append File, Retrieval from Research Log, Append Summary to Research Log, Python REPL, Edit Script Segment (AI)")
    parser.add_argument("--actions-add-to-prompt", type=str, nargs='+', default=[], help="actions to add")
    parser.add_argument("--retrieval", action="store_true", help="enable retrieval")
    parser.add_argument("--valid-format-entries", type=str, nargs='+', default=None, help="valid format entries")
    parser.add_argument("--max-steps-in-context", type=int, default=3, help="max steps in context")
    parser.add_argument("--max-observation-steps-in-context", type=int, default=3, help="max observation steps in context")
    parser.add_argument("--max-retries", type=int, default=5, help="max retries")

    args = parser.parse_args()

    with open("/home/agent/instructions.txt", "r") as f:
        instructions = f.read()
        
    with Environment(args, instructions=instructions) as env:
        
        #env.research_problem = instructions
        #args.instructions = instructions
        print("=====================================")
        #research_problem, benchmark_folder_name = env.get_task_description()
        #print("Benchmark folder name: ", benchmark_folder_name)
        print("Research problem: ", instructions, file=sys.stderr)
        print("Lower level actions enabled: ", [action.name for action in env.low_level_actions], file=sys.stderr)
        print("High level actions enabled: ", [action.name for action in env.high_level_actions], file=sys.stderr)
        print("Read only files: ", env.read_only_files, file=sys.stderr)
        print("=====================================")  

        agent = ResearchAgent(args, env=env)
        final_message = agent.run(env)
        print("=====================================")
        print("Final message: ", final_message)

    env.save("final")


if __name__ == "__main__":
    main()
