import json
import time
import torch
from pathlib import Path
from typing import Optional, Union
from unittest import TestCase
from dotmap import DotMap
from ohlcformer.utils.configuration_utils import \
    find_checkpoint, load_model_configs, load_from_configs, load_from_dir, save_model_configs, load_model


class TestConfigurationUtils(TestCase):
    @staticmethod
    def create_checkpoint_file(file_name: Optional[str] = None,
                               model_folder: Optional[Union[str, Path]] = None,
                               ckpt_folder: Optional[str] = 'checkpoints'
                               ) -> tuple:
        model_folder = str(time.time()) if model_folder is None else model_folder
        path = Path().absolute()
        checkpoint_path = path / model_folder / ckpt_folder / ('checkpoint.ckpt' if file_name is None else file_name)
        checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

        checkpoint = torch.nn.ModuleDict({
            'linear1': torch.nn.Linear(5, 10),
            'linear2': torch.nn.Linear(10, 5),
        })

        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path, checkpoint

    @staticmethod
    def generate_model_config() -> dict:
        return DotMap({
            "model_type": "FNetForFM",
            "dropout_proba": 0.2,
            "intermediate_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "hidden_dropout_prob": 0.1,
            "max_seq_length": 2000,
            "input_size": 4,
            "hidden_size": 64,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "num_layers": 3,
            "num_attention_heads": 4,
            "encoder": "positional",
        })

    @staticmethod
    def create_configs_file(configs: dict, file_name: Optional[str] = 'configs.json') -> Path:
        model_folder = str(time.time())
        path = Path().absolute()
        configs_path = path / model_folder / ('configs.json' if file_name is None else file_name)
        configs_path.parent.mkdir(exist_ok=True, parents=True)

        with open(configs_path, 'w', encoding='utf8') as outfile:
            json.dump(configs, outfile, ensure_ascii=False)

        return configs_path

    @staticmethod
    def remove_file(file_path: Path):
        try:
            file_path.unlink()
            file_path.parent.rmdir()
            file_path.parent.parent.rmdir()
        except OSError:
            return

    def test_load_model(self):
        configs = self.generate_model_config()
        configs_path = self.create_configs_file(configs)

        self.assertRaises(ValueError, load_model)

        model = load_model(DotMap(configs))
        self.assertEqual(model.__class__.__name__, configs['model_type'])

        model = load_model(configs_path=configs_path.parent)
        self.assertEqual(model.__class__.__name__, configs['model_type'])

        model = load_model(configs_path=configs_path.as_posix())
        self.assertEqual(model.__class__.__name__, configs['model_type'])

        model = load_model(model_dir=configs_path.parent)
        self.assertEqual(model.__class__.__name__, configs['model_type'])

        model = load_model(model_dir=configs_path.as_posix())
        self.assertEqual(model.__class__.__name__, configs['model_type'])

    def test_save_model_configs(self):
        configs = self.generate_model_config()
        configs_folder = Path().absolute() / str(time.time())
        configs_folder.mkdir(exist_ok=True, parents=True)

        save_model_configs(configs, configs_folder)
        loaded_configs = load_model_configs(configs_folder)
        self.assertEqual(configs, loaded_configs)
        self.remove_file(configs_folder)

        save_model_configs(configs, configs_folder.as_posix())
        loaded_configs = load_model_configs(configs_folder)
        self.assertEqual(configs, loaded_configs)
        self.remove_file(configs_folder)

        self.assertRaises(TypeError, save_model_configs, None, configs_folder)

        self.assertRaises(FileNotFoundError, save_model_configs, configs, configs_folder / 'test')

        self.remove_file(configs_folder)

    def test_load_model_configs(self):
        configs = self.generate_model_config()
        configs_path = self.create_configs_file(configs)

        loaded_configs = load_model_configs(configs_path.parent)
        self.assertEqual(configs, loaded_configs)

        loaded_configs = load_model_configs(configs_path)
        self.assertEqual(configs, loaded_configs)

        loaded_configs = load_model_configs(configs_path.as_posix())
        self.assertEqual(configs, loaded_configs)

        loaded_configs = load_model_configs(configs_path.parent.as_posix())
        self.assertEqual(configs, loaded_configs)

        self.remove_file(configs_path)

        configs = self.generate_model_config()
        configs_path = self.create_configs_file(configs, file_name='test.json')
        loaded_configs = load_model_configs(configs_path.parent)
        self.assertEqual(loaded_configs, None)

        loaded_configs = load_model_configs(configs_path.parent / 'configs.json')
        self.assertEqual(loaded_configs, None)

        self.remove_file(configs_path)

        checkpoint_path, _ = self.create_checkpoint_file()
        loaded_configs = load_model_configs(checkpoint_path)
        self.assertEqual(loaded_configs, None)
        self.remove_file(checkpoint_path)

    def test_load_from_configs(self):
        configs = self.generate_model_config()

        model = load_from_configs(configs)
        self.assertEqual(model.__class__.__name__, configs['model_type'])

        self.assertRaises(TypeError, load_from_configs, None)

        self.assertRaises(AttributeError, load_from_configs, {})

        configs['model_type'] = 'not_implemented'
        self.assertRaises(KeyError, load_from_configs, configs)

        del configs['model_type']
        self.assertRaises(TypeError, load_from_configs, configs)

    def test_load_from_dir(self):
        configs = self.generate_model_config()
        configs_path = self.create_configs_file(configs)

        model = load_from_dir(configs_path.parent)
        self.assertEqual(model.__class__.__name__, configs['model_type'])

        model = load_from_dir(configs_path.parent.as_posix())
        self.assertEqual(model.__class__.__name__, configs['model_type'])

        model = load_from_dir(configs_path)
        self.assertEqual(model.__class__.__name__, configs['model_type'])

        self.assertRaises(FileNotFoundError, load_from_dir, 'test')

        self.assertRaises(TypeError, load_from_dir, None)

        checkpoint_path, checkpoint = self.create_checkpoint_file(model_folder=configs_path.parts[-2])
        model = load_from_dir(configs_path.parent.as_posix())
        self.assertEqual(model.__class__.__name__, configs['model_type'])

        self.remove_file(checkpoint_path)
        self.remove_file(configs_path)

        configs = self.generate_model_config()
        configs['model_type'] = 'not_implemented'
        configs_path = self.create_configs_file(configs)
        self.assertRaises(KeyError, load_from_dir, configs_path.parent)
        self.remove_file(configs_path)

        configs = self.generate_model_config()
        del configs['model_type']
        configs_path = self.create_configs_file(configs)
        self.assertRaises(TypeError, load_from_dir, configs_path.parent)
        self.remove_file(configs_path)

    def test_find_checkpoint(self):
        checkpoint_path, checkpoint = self.create_checkpoint_file()
        found = find_checkpoint(checkpoint_path.parent.parent)
        self.assertEqual(checkpoint.__str__(), torch.load(found).__str__())
        self.remove_file(checkpoint_path)

        checkpoint_path, checkpoint = self.create_checkpoint_file(file_name='test.ckpt')
        found = find_checkpoint(checkpoint_path.parent.parent)
        self.assertEqual(checkpoint.__str__(), torch.load(found).__str__())
        self.remove_file(checkpoint_path)

        checkpoint_path, checkpoint = self.create_checkpoint_file(file_name='test.ckpt')
        found = find_checkpoint(checkpoint_path.parent.parent.as_posix())
        self.assertEqual(checkpoint.__str__(), torch.load(found).__str__())
        self.remove_file(checkpoint_path)

        checkpoint_path, checkpoint = self.create_checkpoint_file(file_name='test')
        found = find_checkpoint(checkpoint_path.parent.parent)
        self.assertEqual(found, None)
        self.remove_file(checkpoint_path)

        checkpoint_path, checkpoint = self.create_checkpoint_file()
        self.remove_file(checkpoint_path)
        found = find_checkpoint(checkpoint_path.parent.parent)
        self.assertEqual(found, None)

        checkpoint_path, checkpoint = self.create_checkpoint_file(file_name='test.ckpt')
        found = find_checkpoint(checkpoint_path.parent)
        self.assertEqual(found, None)
        self.remove_file(checkpoint_path)
