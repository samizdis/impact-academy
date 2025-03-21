from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import torch

# Load the dataset
dataset = load_from_disk("/home/agent/resources/raft/tai_safety_research")

# Load the tokenizer and model
model_name = "bigscience/mt0-large"

tokenizer = AutoTokenizer.from_pretrained("/home/agent/resources/mt0-large/tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("/home/agent/resources/mt0-large/model")

class_names = ["TAI safety research", "not TAI safety research"]

# function to classify text
def classify_text(get_prompt, text):
    prompt = get_prompt(text, class_names)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=8, num_return_sequences=1)
    
    classification = tokenizer.decode(outputs[0], skip_special_tokens=True)
    classification = classification.split(" Classification:")[-1].strip()
    
    # if any class name matches with classification, return the corresponding class name
    # otherwise, return "unknown"
    for class_name in class_names:
        if class_name == classification:
            return class_name
    else:
        return "unknown"

sample = dataset["train"].shuffle()

def calculate_accuracy(get_prompt):
    results = []
    for item in sample:
        abstract = item["Abstract Note"]
        true_label = class_names[0] if item["Label"] == 1 else class_names[1]
        predicted_label = classify_text(get_prompt, abstract)
        result = {
            "Abstract": abstract,
            "True Label": true_label.strip(),
            "Predicted Label": predicted_label.strip()
        }
        results.append(result)

        print(f"Abstract: {result['Abstract'][:100]}...")
        print(f"True Label: {result['True Label']}")
        print(f"Predicted Label: {result['Predicted Label']}\n\n")

    correct_predictions = sum(1 for result in results if result["True Label"] == result["Predicted Label"])
    accuracy = correct_predictions / len(results)
    return accuracy