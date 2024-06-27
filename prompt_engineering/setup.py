from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

dataset = load_dataset("ought/raft", "tai_safety_research")
dataset.save_to_disk("resources/raft/tai_safety_research")

model_name = "bigscience/bloom-560m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("resources/bloom-560m/tokenizer")

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained("resources/bloom-560m/model")


