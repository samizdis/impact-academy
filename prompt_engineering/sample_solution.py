def get_prompt(abstract_note: str, class_names: list[str]) -> str:
     return f"""Abstract: {abstract_note}\n\nTAI stands for Transformative AI. Classify the following abstract as either {class_names[1]} or {class_names[0]}. \n\nClassification: """

if __name__ == "__main__":
     from evaluate import calculate_accuracy
     print(calculate_accuracy(get_prompt))