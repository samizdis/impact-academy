def get_prompt(abstract_note: str, class_names: list[str]) -> str:
     return f"""{abstract_note}\n\nClassification: """

if __name__ == '__main__':
     from evaluate import calculate_accuracy
     print(calculate_accuracy(get_prompt))