# from utils import get_model_and_tokenizer
import torch
from transformers import AdamW
import tqdm
from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM

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

def get_data(forget_corpous, retain_corpous, min_len=50, max_len=2000, batch_size=4):
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
        else:
            for line in open(f"data/{name}.jsonl", "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)["text"]
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (get_dataset(forget_corpous), get_dataset(retain_corpous))

def run_rmu(
    model_name_or_path,
    updated_model,
    frozen_model,
    tokenizer,
    forget_data,
    retain_data,
    layer_id,
    layer_ids,
    param_ids,
    lr,
    steering_coeff,
    alpha,
    max_num_batches,
    module_str,
    epochs
):

    updated_model = updated_model.train()
    frozen_model = frozen_model.eval()
    params = get_params(updated_model, layer_ids, param_ids)
    if params == []:
        print("No parameters found")
        return
    else:
        print(f"Found {len(params)} parameters")
    optimizer = AdamW(params, lr=lr)
    frozen_module = eval(
        module_str.format(model_name="frozen_model", layer_id=layer_id)
    )
    updated_module = eval(
        module_str.format(model_name="updated_model", layer_id=layer_id)
    )
    
    random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
    control_vec = (random_vector / torch.norm(random_vector)) * steering_coeff

    num_batches = min(
        max_num_batches,
        len(forget_data),
        len(retain_data),
    )
    
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    for epoch in range(epochs):
        print(f"======= Epoch {epoch} =======")
        train_loss_forget = 0.0
        train_loss_retain = 0.0
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                unlearn_batch = forget_data[idx]
                retain_batch = retain_data[idx]

                # Unlearning loss
                max_length = 512
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(updated_model.device)
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)

                forget_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec
                )
                train_loss_forget += forget_loss.item()

                # Retain loss
                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(updated_model.device)
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                train_loss_retain += retain_loss.item()

                # Update model
                loss = forget_loss + alpha * retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item():.4g} | unlearn_loss: {forget_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
                
                pbar.update(1)

        # train_loss = train_loss_forget + alpha * train_loss_retain
        
        # mmlu_accuracy, wmdp_accuracy = get_val_acc(updated_model, tokenizer)
        # print(f"train_loss: {train_loss:.4g} | mmlu_accuracy: {mmlu_accuracy:.4g} | wmdp_accuracy: {wmdp_accuracy:.4g}")

    tokenizer.truncation_side = truncation_side
    # Save model
    path = f"models/{model_name_or_path}-unlearn"
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")

def get_val_acc(model, tokenizer):
    results = simple_evaluate(
        HFLM(model, tokenizer=tokenizer),
        tasks=["wmdp", "mmlu"],
        device="cuda:0",
        limit=0.2,
        random_seed = 0,
        numpy_random_seed = 0,
        torch_random_seed = 0,
        fewshot_random_seed = 0
    )
    wmdp_accuracy = results["results"]["wmdp"]["acc,none"]
    mmlu_accuracy = results["results"]["mmlu"]["acc,none"]
    
    return mmlu_accuracy, wmdp_accuracy

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

def main():
    model_name_or_path = "Meta-Llama-3-8B"
    hf_repo_id = f"meta-llama/{model_name_or_path}"
    updated_model, tokenizer = get_model_and_tokenizer(hf_repo_id)
    frozen_model, _ = get_model_and_tokenizer(hf_repo_id)

    print(f"Updated model {updated_model.get_memory_footprint()/1e9}")
    print(f"Frozen model {frozen_model.get_memory_footprint()/1e9}")
    
    module_str = "{model_name}.model.layers[{layer_id}]"

    forget_data, retain_data = get_data(
        "cyber-forget-corpus", "wikitext"
    )
    
    run_rmu(
        model_name_or_path=model_name_or_path,
        updated_model=updated_model,
        frozen_model=frozen_model,
        tokenizer=tokenizer,
        forget_data=forget_data,
        retain_data=retain_data,
        layer_id=11,
        layer_ids=[9, 10, 11],
        param_ids=[6],
        lr=0.0000459,
        steering_coeff=188.565,
        alpha=1816,
        max_num_batches=200,
        module_str=module_str,
        epochs=3
    )
    
main()