import yaml
import pandas as pd
import os


def rasa_data_to_csv(input_file_path: str, output_file_path: str):
    with open(input_file_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    nlu_data = data['nlu']
    nlu = []
    for intent in nlu_data:
        if "intent" not in intent or "examples" not in intent:
            continue
        intent_name = intent['intent']
        examples = [i.lstrip('- ') for i in intent['examples'].split('\n')]
        for example in examples:
            nlu.append({'utterance_example': example, 'expected_intent': intent_name})
    
    if output_file_path:
        print(f'Writing to {output_file_path}')
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        pd.DataFrame(nlu).to_csv(output_file_path, index=False, header=False)
    else:
        return pd.DataFrame(nlu)