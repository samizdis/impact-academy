from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

dataset = load_dataset("ought/raft", "tai_safety_research")
dataset.save_to_disk("assets/raft/tai_safety_research")

model_name = "bigscience/mt0-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("assets/mt0-large/tokenizer")

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.save_pretrained("assets/mt0-large/model")
