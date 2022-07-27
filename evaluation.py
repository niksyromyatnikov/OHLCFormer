from pathlib import Path
from typing import Union

from dotmap import DotMap
from heapq import nsmallest, nlargest
from . import load_model
from .losses import get_metric_direction
from .models import ModelForFM


def run_tests(tests: dict,
              model: ModelForFM = None,
              configs: DotMap = None,
              configs_path: Union[str, Path] = None,
              model_dir: Union[str, Path] = None
              ) -> dict:
    model = model if model is not None else load_model(configs, configs_path, model_dir)

    if model is None:
        raise ValueError('Expected one of [model object, configs, configs_path, model_dir] to be provided')

    result = {}
    default_model_configs = model.get_configs()

    trainer = model.configure_trainer()

    for name, test in tests.items():
        test_configs = {}

        if isinstance(test, str) or isinstance(test, Path):
            test_configs['dataset_path'] = test
        elif isinstance(test, dict):
            for config_name, config_val in test.items():
                if config_name == 'dataset_path':
                    test_configs['dataset_path'] = config_val
                    continue
                try:
                    model.set_config(config_name, config_val)
                except TypeError as e:
                    print(f'Failed to set config {config_name} to {config_val}')
                    print(e)

        model.load_dataset({'test_dataset': test_configs['dataset_path']})
        result[name] = trainer.test(model, verbose=False)
        model.set_configs(default_model_configs)

    return result


class TestResult(object):
    def __init__(self, score: float, model_name: str):
        self.score = score
        self.model_name = model_name

    def __lt__(self, other):
        return self.score < other.score

    def __str__(self):
        return f'{self.model_name}: {self.score}'

    def __repr__(self):
        return f'{self.model_name}: {self.score}'


def compare_models(tests: dict, models: dict, top_k=0) -> (dict, dict):
    if not tests or not models:
        raise ValueError('Expected at least one test and one model!')

    top_k = len(models) if top_k <= 0 else top_k
    results = {name: {} for name in tests.keys()}
    top = {name: {} for name in tests.keys()}

    for model_name, model in models.items():
        tests_result = run_tests(tests, **model)

        for test_name, test_result in tests_result.items():
            if not test_result:
                continue

            result = test_result[0]

            for metric_name, score in result.items():
                if results[test_name].get(metric_name, None) is None:
                    results[test_name][metric_name] = []
                results[test_name][metric_name].append(TestResult(score, model_name))

    for test_name in top.keys():
        top[test_name] = {
            metric_name: nsmallest(top_k, results[test_name][metric_name]) if not get_metric_direction(metric_name)
            else nlargest(top_k, results[test_name][metric_name]) for metric_name in results[test_name].keys()
        }

    return (top, results) if top_k != len(models) else top
