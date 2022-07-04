import os
import json


def save_model_configs(configs, model_dir: str):
    with open(model_dir + 'configs.json', 'w', encoding='utf8') as json_file:
        json.dump(configs, json_file, indent=2)


def load_model_configs(model_dir: str):
    files = os.listdir(model_dir)
    configs = None

    if 'configs.json' in files:
        with open(model_dir + 'configs.json', 'r', encoding='utf8') as json_file:
            configs = json.load(json_file)

    return configs
