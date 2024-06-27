import evaluate

def get_prompt(abstract_note: str, class_names: list[str]) -> str:
     return f"""Classifty the following abstract as either '{class_names[0]}' or '{class_names[1]}'. 
            \n{abstract_note}\n
            Classification: 
            """

if __name__ == '__main__':
    accuracy = evaluate.calculate_accuracy(get_prompt)
    print(f"{accuracy:.2f}")