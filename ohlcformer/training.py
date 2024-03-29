from pathlib import Path
from typing import Union
from dotmap import DotMap
from ohlcformer import logging
from ohlcformer.models import ModelForFM
from ohlcformer.utils import load_model

logger = logging.get_logger(__name__)


def run_training(model: ModelForFM = None,
                 configs: DotMap = None,
                 configs_path: Union[str, Path] = None,
                 model_dir: Union[str, Path] = None,
                 datasets: dict = None,
                 evaluate: bool = False,
                 verbose: bool = True,
                 ) -> dict:
    eval_result = {}

    logger.info('Running model training.')

    for name, dataset in datasets.items():
        train_configs = {}

        model = model if model is not None else load_model(configs, configs_path, model_dir)

        if model is None:
            raise ValueError('Expected one of [model object, configs, configs_path, model_dir] to be provided')

        configs = model.get_configs()

        if isinstance(dataset, str) or isinstance(dataset, Path):
            train_configs['dataset_path'] = dataset
            dataset = train_configs

        if isinstance(dataset, dict):
            for config_name, config_val in dataset.items():
                try:
                    model.set_config(config_name, config_val)
                except TypeError as e:
                    logger.error(f'Failed to set config {config_name} to {config_val}', exc_info=True)
                    logger.error(e, exc_info=True)

        model.load_dataset()

        trainer_configs = model.get_configs().get('trainer', {})

        if model_dir is not None:
            trainer_configs['default_root_dir'] = model_dir
        elif trainer_configs.get('default_root_dir', None) is None:
            raise ValueError('Expected model_dir or trainer.default_root_dir to be provided')

        trainer_configs['default_root_dir'] = Path(trainer_configs['default_root_dir']) / name

        trainer = model.configure_trainer(**trainer_configs)
        trainer.fit(model)

        if evaluate:
            logger.info(f'Running model evaluation on {name} dataset.')
            eval_result[name] = trainer.test(model, verbose=verbose)
            logger.debug(f'{name} evaluation result: {eval_result[name]}')

        model = None

    if evaluate:
        return eval_result
