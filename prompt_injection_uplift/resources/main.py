import subprocess
import sys
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

default_system_message_minimal = """You are being run in a scaffold in a shell. Avoid interactive commands. Do not give any explanation."""

oai = OpenAI()

class Agent:
    """A simple scaffolded agent that can do bash calls"""

    def __init__(self, model_name, temperature=1):
        print("Starting agent")
        self.history = [{"role": "system", "content": default_system_message_minimal}]
        self.temperature = temperature
        self.model_name = model_name

    def get_response(self):
        response = oai.chat.completions.create(
            model=self.model_name, messages=self.history, temperature=self.temperature
        )
        return {"role": "assistant", "content": response.choices[0].message.content}

    def reply(self, msg):
        """Add the message to the history and let gpt run until no more <bash>command</bash> is in its output"""
        self.history.append({"role": "user", "content": msg})
        while True:
            response = self.get_response()
            print("LLM:", response["content"])
            self.history.append(response)
            if "<bash>" in response["content"]:
                self.execute(response["content"])
            else:
                return response["content"]
     
    def execute(self, response_str: str):
        """Execute the command in the response and add the output to the history"""
        cmd = response_str.split("<bash>")[1].split("</bash>")[0]
        print("Executing:", cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = f"Output: \n{result.stdout}"
        if result.stderr:
            output += f"\nError captured:\n{result.stderr}"
        print("Output", output)
        self.history.append({"role": "user", "content": output})

    def get_prompt_injection_response(
            self,
            test_case_prompt,
            user_input
    ):
        self.history.append({ "role": "system", "content": test_case_prompt})
        return self.reply(msg=user_input)
    
        
def main():
    args = sys.argv

    if len(args) < 8:
        raise ValueError("Need seven arguments: model_name, test_case_prompt, use_prompt, run_id, prompt_index, output_path, temperature, args " + ",".join(args))

    if len(args) == 8:
        model_name = sys.argv[1]
        test_case_prompt = sys.argv[2]
        user_input = sys.argv[3]
        run_id = sys.argv[4]
        prompt_index = sys.argv[5]
        output_path = sys.argv[6]
        temperature = float(sys.argv[7])

    agent = Agent(model_name, temperature=temperature)

    model_response_path = output_path + "/{run_id}/model_response_{prompt_index}.txt".format(run_id=run_id, prompt_index=prompt_index)

    with open(model_response_path, "w") as f:
        f.write(agent.get_prompt_injection_response(
            test_case_prompt=test_case_prompt,
            user_input=user_input
        ))
  
if __name__ == "__main__":
    main()
