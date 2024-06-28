import evaluate

def get_prompt(abstract_note: str, class_names: list[str]) -> str:
     return f"""{abstract_note}\n\nClassification: """

if __name__ == '__main__':
    accuracy = evaluate.calculate_accuracy(get_prompt)
    print(f"{accuracy:.2f}")