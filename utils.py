import os
import json
from pathlib import Path
from typing import Union

from .models import Model
from .models.builder import ModelBuilder
from dotmap import DotMap


def load_model(configs: DotMap = None, configs_path: str = None, model_dir: str = None) -> Model:
    configs = load_model_configs(configs_path) if configs is None and configs_path is not None else configs

    if configs is not None:
        model = load_from_configs(configs)
    elif model_dir is not None:
        model = load_from_dir(model_dir)
    else:
        raise ValueError('Expected one of [model configs, configs_path, model_dir] to be provided')

    return model


def save_model_configs(configs, model_dir: Union[str, Path]):
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)

    with open(model_dir / 'configs.json', 'w', encoding='utf8') as json_file:
        json.dump(configs, json_file, indent=2)


def load_model_configs(path: Union[str, Path]):
    configs = None

    if not isinstance(path, Path):
        path = Path(path)

    try:
        if path.is_dir() and 'configs.json' in os.listdir(path):
            path /= 'configs.json'
        if not path.is_file():
            raise FileNotFoundError('No configs found in ' + str(path))

        with open(path, 'r', encoding='utf8') as json_file:
            configs = DotMap(json.load(json_file))
    except Exception as e:
        print(e)

    return configs


def load_from_configs(configs):
    if configs is None:
        raise TypeError('Expected configs object but got None instead')

    model = ModelBuilder.build(configs)

    return model


def load_from_dir(model_dir: Union[str, Path]):
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)

    configs = load_model_configs(model_dir)

    if configs is None:
        raise FileNotFoundError('No configs found in ' + model_dir.as_posix())

    checkpoint = find_checkpoint(model_dir)
    if checkpoint is not None:
        configs.checkpoint = checkpoint
        print("Loading model from checkpoint: ", configs.checkpoint)

    model = ModelBuilder.build(configs)

    return model


def find_checkpoint(model_dir: Union[str, Path]):
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)
    checkpoint_dir = model_dir / 'checkpoints/'

    checkpoint = None

    try:
        for file in os.listdir(checkpoint_dir):
            file = str(file)
            if file.endswith('.ckpt'):
                checkpoint = checkpoint_dir / file
    except Exception as e:
        print(e)

    return checkpoint
