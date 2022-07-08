import pytorch_lightning as pl
from dotmap import DotMap
from losses import get_metric_direction
from utils import load_model_configs, load_from_configs, load_from_dir
from heapq import nsmallest, nlargest


def run_tests(tests: dict, model=None, configs: DotMap = None, configs_path: str = None, model_dir: str = None) -> dict:
    configs = load_model_configs(configs_path) if configs is None and configs_path is not None else configs

    if model is None:
        if configs is not None:
            model = load_from_configs(configs)
        elif model_dir is not None:
            model = load_from_dir(model_dir)
        else:
            raise ValueError('Expected one of [model object, configs, configs_path, model_dir] to be provided')

    model.cuda()

    result = {}

    trainer = pl.Trainer(gpus=1)

    for name, test_path in tests.items():
        model.load_dataset({'test_dataset': test_path})
        # print(name, len(model.test_loader))
        result[name] = trainer.test(model, verbose=False)

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

    return top, results
