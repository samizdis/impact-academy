from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import torch

# Load the tokenizer and model
model_name = "bigscience/bloom-560m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class_names = ["TAI safety research", "not TAI safety research"]

# function to classify text
def classify_text(get_prompt, text):
    prompt = get_prompt(text, class_names)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, num_return_sequences=1)
    

    classification = tokenizer.decode(outputs[0], skip_special_tokens=True)
    classification = classification.split(" Classification:")[-1].strip()
    
    print("-----------------------------------------------")
    print(f"Model's classification as-is: {classification}")
    print("-----------------------------------------------")

    # if any class name matches with classification, return the corresponding class name
    # otherwise, return "unknown"
    for class_name in class_names:
        if class_name == classification:
            return class_name
    else:
        return "unknown"

def get_test_set():
    import json
    with open('labeled_examples.json', 'r') as f:
        labeled_examples = json.load(f)
    return labeled_examples

def get_baseline_prompt(text, class_names):
    return f"""{text}\n\nClassification: """

def calculate_accuracy(get_prompt=get_baseline_prompt):
    results = []
    labeled_examples = get_test_set()
    for item in labeled_examples:
        abstract = item["Abstract Note"]
        true_label = item["Label"]
        predicted_label = classify_text(get_prompt, abstract)
        results.append({
            "Abstract": abstract,
            "True Label": true_label.strip(),
            "Predicted Label": predicted_label.strip()
        })

    # Print results
    for i, result in enumerate(results, 1):
        print(f"Sample {i}:")
        print(f"Abstract: {result['Abstract'][:100]}...")  # Print first 100 characters
        print(f"True Label: {result['True Label']}")
        print(f"Predicted Label: {result['Predicted Label']}")
        print()


    # Calculate accuracy
    correct_predictions = sum(1 for result in results if result["True Label"] == result["Predicted Label"])
    accuracy = correct_predictions / len(results)
    return accuracy
