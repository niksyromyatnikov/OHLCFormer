from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import DataProcessor, mask_tokens
from losses import LossBuilder, MaskedDirectionLoss, default_loss
from models.modeling import Model


class ModelForFM(pl.LightningModule):

    def __init__(self, configs):
        super(ModelForFM, self).__init__()

        self.configs = configs

        self.model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = Model(configs, self.model_device)

        self.losses = []
        for loss_config in configs.get('losses', default_loss):
            self.losses.append(LossBuilder.build(loss_config))

        self.data_processor = DataProcessor(configs)
        self.metrics = configs.get('metrics', [])

        self.load_checkpoint(configs)
        self.load_weights(configs)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def forward(self, **params):
        device_params = {k: v.to(self.model_device) if v is not None else v for k, v in params.items()}

        outputs = self.net(**device_params)

        return outputs

    def prepare_batch(self, batch) -> tuple:
        if not self.configs.get('lazy_preprocessing', False):
            return batch

        input_ids, attention_mask = batch[:2]

        args = {'input_ids': input_ids, 'attention_mask': attention_mask}

        if self.configs.get('max_seq_length', None) is not None:
            args['seq_len'] = self.configs['max_seq_length']
        if self.configs.get('mask_proba', None) is not None:
            args['mask_proba'] = self.configs['mask_proba']
        if self.configs.get('prediction_len', None) is not None:
            args['prediction_len'] = self.configs['prediction_len']

        input_ids, labels, mask = mask_tokens(**args)

        return [input_ids, attention_mask, mask, labels] + batch[2:]

    def training_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, mask, labels = self.prepare_batch(batch)[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        pooler_output = outputs[1]

        # print(pooler_output, pooler_output.shape)

        return self.calculate_losses(pooler_output, labels, mask)

    def validation_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, mask, labels = self.prepare_batch(batch)[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        pooler_output = outputs[1]

        return self.calculate_losses(pooler_output, labels, mask, stage='val')

    def validation_epoch_end(self, outputs):
        avg_loss, avg_metrics = self.aggregate_metrics(outputs, stage='val')

        tensorboard_logs = {**{'avg_val_loss': avg_loss}, **avg_metrics}

        self.log_dict(tensorboard_logs, prog_bar=True)

        # return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, mask, labels = self.prepare_batch(batch)[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        pooler_output = outputs[1]

        return self.calculate_losses(pooler_output, labels, mask, stage='test')

    def test_epoch_end(self, outputs):
        avg_loss, avg_metrics = self.aggregate_metrics(outputs, stage='test')

        tensorboard_logs = {**{'avg_test_loss': avg_loss}, **avg_metrics}

        self.log_dict(tensorboard_logs, prog_bar=True)

        # return {'avg_test_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad],
                                lr=self.configs.optimizer.learning_rate,
                                eps=self.configs.optimizer.epsilon)

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        return self.test_loader

    def load_dataset(self, dataset_path=None) -> tuple:
        dataset_path = dataset_path if dataset_path is not None else self.configs.dataset_path

        dataset = self.data_processor.prepare_dataset(dataset_path, self.configs.train_set_prop,
                                                      self.configs.val_set_prop, self.configs.test_set_prop,
                                                      self.configs.batch_size)

        self.train_loader = dataset[0] if dataset[0] is not None else self.train_loader
        self.val_loader = dataset[1] if dataset[1] is not None else self.val_loader
        self.test_loader = dataset[2] if dataset[2] is not None else self.test_loader

        return self.train_loader is not None, self.val_loader is not None, self.test_loader is not None

    def calculate_losses(self, output: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, stage: str = '') -> dict:
        calculated_losses = {}

        for loss in self.losses:
            name = (stage + '_' if stage is not None and len(stage) > 0 else '') + loss.name
            if isinstance(loss, MaskedDirectionLoss):
                loss_val, f1 = loss(output.detach(), labels.to(self.model_device), mask.to(self.model_device),
                                    return_f1=True)
                calculated_losses[name] = loss_val
                calculated_losses['_'.join(loss.name.split('_')[:-1] + ['f1'])] = f1
            else:
                calculated_losses[name] = loss(output, labels.to(self.model_device), mask.to(self.model_device))
                calculated_losses[name + '_mean'] = loss(output, labels.to(self.model_device),
                                                         mask.to(self.model_device), reduction='mean')

        return calculated_losses

    def aggregate_metrics(self, outputs: list, stage='val'):
        avg_loss = torch.stack([x[stage + '_loss'] for x in outputs]).mean()
        avg_metrics = defaultdict(list)

        for x in outputs:
            for k, v in x.items():
                key = 'avg_' + k
                if isinstance(v, torch.Tensor):
                    avg_metrics[key].append(v)
                else:
                    avg_metrics[key].append(torch.tensor(v))

        for k, v in avg_metrics.items():
            avg_metrics[k] = torch.stack(v).mean()

        return avg_loss, avg_metrics

    def load_checkpoint(self, configs):
        if configs.get('checkpoint', None) is not None:
            try:
                self.load_state_dict(torch.load(configs.checkpoint)['state_dict'])
            except Exception as e:
                print(e)
                print('Failed to load checkpoint from {}'.format(configs.checkpoint))

    def load_weights(self, configs):
        if configs.get('weights_path', None) is not None:
            try:
                self.load_state_dict(torch.load(configs.weights_path))
            except Exception as e:
                print(e)
                print('Failed to load weights from {}'.format(configs.weights_path))
