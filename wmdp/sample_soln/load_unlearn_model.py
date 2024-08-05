from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

m = AutoModelForCausalLM.from_pretrained("saisasanky/Meta-Llama-3-8B-unlearn", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

print(m.get_memory_footprint()/1e9)


tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
tokenizer.mask_token_id = tokenizer.eos_token_id
tokenizer.sep_token_id = tokenizer.eos_token_id
tokenizer.cls_token_id = tokenizer.eos_token_id

m.save_pretrained("models/Meta-Llama-3-8B-unlearn")
tokenizer.save_pretrained("models/Meta-Llama-3-8B-unlearn")