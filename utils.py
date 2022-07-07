import os
import json
from models.builder import ModelBuilder
from dotmap import DotMap


def save_model_configs(configs, model_dir: str):
    model_dir += '/' if not model_dir.endswith('/') else ''
    with open(model_dir + 'configs.json', 'w', encoding='utf8') as json_file:
        json.dump(configs, json_file, indent=2)


def load_model_configs(model_dir: str):
    configs = None

    model_dir += '/' if not model_dir.endswith('/') else ''

    try:
        if 'configs.json' in os.listdir(model_dir):
            with open(model_dir + 'configs.json', 'r', encoding='utf8') as json_file:
                configs = DotMap(json.load(json_file))
    except Exception as e:
        print(e)

    return configs


def load_from_configs(configs):
    if configs is None:
        raise TypeError('Expected configs object but got None instead')

    model = ModelBuilder.build(configs)

    return model


def load_from_dir(model_dir: str):
    model_dir += '/' if not model_dir.endswith('/') else ''
    configs = load_model_configs(model_dir)

    if configs is None:
        raise FileNotFoundError('No configs found in ' + model_dir)

    checkpoint = find_checkpoint(model_dir)
    if checkpoint is not None:
        configs.checkpoint = checkpoint
        print("Loading model from checkpoint: ", configs.checkpoint)

    model = ModelBuilder.build(configs)

    return model


def find_checkpoint(model_dir: str):
    checkpoint = None
    model_dir += '/' if not model_dir.endswith('/') else ''
    checkpoint_dir = model_dir + 'checkpoints/'

    try:
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.ckpt'):
                checkpoint = checkpoint_dir + file
    except Exception as e:
        print(e)

    return checkpoint
