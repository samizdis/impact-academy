import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login(new_session=False)

def read_configuration():
    with open('/home/ubuntu/resources/task_config.pkl', 'rb') as f:
        task_config = pickle.load(f)
    return task_config

def get_model_and_tokenizer():
    task_config = read_configuration()
    model = AutoModelForCausalLM.from_pretrained(task_config['model']['hf_repo_id'])
    tokenizer = AutoTokenizer.from_pretrained(task_config['model']['hf_repo_id'])

    return model, tokenizer