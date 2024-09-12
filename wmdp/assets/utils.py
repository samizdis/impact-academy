import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset
import torch

login(new_session=False)

def read_configuration():
    with open('/home/ubuntu/resources/task_config.pkl', 'rb') as f:
        task_config = pickle.load(f)
    return task_config

def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]

def get_data(forget_corpous, retain_corpous, min_len=50, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x["text"]) > min_len:
                    data.append(str(x["text"]))
        elif name == "cyber-forget-corpus":
            raw_data = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train")
            for x in raw_data:
                if len(x["text"]) > min_len:
                    data.append(str(x["text"]))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (get_dataset(forget_corpous), get_dataset(retain_corpous))

def get_model_and_tokenizer(hf_repo_id):
    # load into gpu
    model = AutoModelForCausalLM.from_pretrained(hf_repo_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer