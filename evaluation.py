import pytorch_lightning as pl
from dotmap import DotMap

from model import ModelForFM
from utils import load_model_configs


def run_tests(tests: dict, configs: DotMap = None, configs_path: str = None) -> dict:
    configs = load_model_configs(configs_path) if configs is None else configs

    model = ModelForFM(configs)
    model.cuda()

    result = {}

    for name, test_path in tests.items():
        model.load_dataset({'test_dataset': test_path})
        # print(name, len(model.test_loader))
        trainer = pl.Trainer(gpus=1)
        result[name] = trainer.test(model, verbose=False)

    return result
