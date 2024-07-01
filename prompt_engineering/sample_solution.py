def get_prompt(abstract_note: str, class_names: list[str]) -> str:
     return f"""TAI stands for Transformative AI. I will classify the following abstract as either {class_names[1]} or {class_names[0]}. \n\nAbstract: {abstract_note}\n\n I will now output just one of the class names and nothing else. Classification: """