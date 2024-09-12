import torch
from transformers import AdamW
import tqdm
from utils import get_data, get_params, forward_with_cache, get_model_and_tokenizer

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

                max_length = 512
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(updated_model.device)
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)

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
                    frozen_retain_activations, updated_retain_activations
                )

                forget_loss = torch.nn.functional.mse_loss(
                    frozen_retain_activations, control_vec
                )
                train_loss_forget += forget_loss.item()


                train_loss_retain += retain_loss.item()

                # Update model
                loss = forget_loss + alpha * retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item():.4g} | unlearn_loss: {forget_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
                
                pbar.update(1)

    tokenizer.truncation_side = truncation_side
    # Save model
    path = f"models/{model_name_or_path}-unlearn"
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")

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