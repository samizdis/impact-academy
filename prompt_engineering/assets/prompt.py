def get_prompt(abstract_note: str, class_names: list[str]) -> str:
     return f"""{abstract_note}\n\nClassification: """