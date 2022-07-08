import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data import DataProcessor
from losses import MaskedMSELoss, MaskedRMSELoss, MaskedMAELoss, MaskedDirectionLoss
from models.modeling import Model


class ModelForFM(pl.LightningModule):

    def __init__(self, configs):
        super(ModelForFM, self).__init__()

        self.configs = configs

        self.model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = Model(configs, self.model_device)

        self.criterion = MaskedMSELoss(reduction=configs.get('reduction_strategy', 'sum'))
        self.rmse = MaskedRMSELoss(reduction=configs.get('reduction_strategy', 'sum'))
        self.mae = MaskedMAELoss(reduction=configs.get('reduction_strategy', 'sum'))
        self.mdl = MaskedDirectionLoss(reduction=configs.get('reduction_strategy', 'sum'))

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

    def mask_input(self, inputs, mask):
        # masking input
        mask = torch.ones(inputs.shape[0], inputs.shape[1]).to(self.model_device)
        mask[inputs == 0] = 0

        inputs[mask == 0] = torch.Tensor([0, 0, 0, 0])

        return mask

    def training_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, mask, labels = batch[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        pooler_output = outputs[1]

        # print(pooler_output, pooler_output.shape)

        loss = self.criterion(pooler_output, labels.to(self.model_device), mask.to(self.model_device))
        loss_mean = self.criterion(pooler_output, labels.to(self.model_device), mask.to(self.model_device),
                                   reduction='mean')

        rmse_loss = self.rmse(pooler_output, labels.to(self.model_device), mask.to(self.model_device))
        rmse_loss_mean = self.rmse(pooler_output, labels.to(self.model_device), mask.to(self.model_device),
                                   reduction='mean')

        mae_loss = self.mae(pooler_output, labels.to(self.model_device), mask.to(self.model_device))
        mae_loss_mean = self.mae(pooler_output, labels.to(self.model_device), mask.to(self.model_device),
                                 reduction='mean')

        mdl_loss, mdl_f1 = self.mdl(pooler_output.detach(), labels.to(self.model_device), mask.to(self.model_device),
                                    return_f1=True)

        return {'loss': loss, 'loss_mean': loss_mean, 'rmse_loss': rmse_loss, 'rmse_loss_mean': rmse_loss_mean,
                'mae_loss': mae_loss, 'mae_loss_mean': mae_loss_mean, 'mask_direction_loss': mdl_loss,
                'mask_direction_f1': mdl_f1}

    def validation_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, mask, labels = batch[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        pooler_output = outputs[1]

        loss = self.criterion(pooler_output, labels.to(self.model_device), mask.to(self.model_device))
        loss_mean = self.criterion(pooler_output, labels.to(self.model_device), mask.to(self.model_device),
                                   reduction='mean')

        rmse_loss = self.rmse(pooler_output, labels.to(self.model_device), mask.to(self.model_device))
        rmse_loss_mean = self.rmse(pooler_output, labels.to(self.model_device), mask.to(self.model_device),
                                   reduction='mean')

        mae_loss = self.mae(pooler_output, labels.to(self.model_device), mask.to(self.model_device))
        mae_loss_mean = self.mae(pooler_output, labels.to(self.model_device), mask.to(self.model_device),
                                 reduction='mean')

        mdl_loss, mdl_f1 = self.mdl(pooler_output.detach(), labels.to(self.model_device), mask.to(self.model_device),
                                    return_f1=True)

        return {'val_loss': loss, 'val_loss_mean': loss_mean, 'val_rmse_loss': rmse_loss,
                'val_rmse_loss_mean': rmse_loss_mean, 'val_mae_loss': mae_loss, 'val_mae_loss_mean': mae_loss_mean,
                'val_mask_direction_loss': mdl_loss, 'val_mask_direction_f1': mdl_f1}

    def validation_epoch_end(self, outputs):
        avg_loss, avg_metrics = self.aggregate_metrics(outputs, stage='val')

        tensorboard_logs = {**{'avg_val_loss': avg_loss}, **avg_metrics}

        self.log_dict(tensorboard_logs, prog_bar=True)

        # return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, mask, labels = batch[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        pooler_output = outputs[1]

        loss = self.criterion(pooler_output, labels.to(self.model_device), mask.to(self.model_device))
        loss_mean = self.criterion(pooler_output, labels.to(self.model_device), mask.to(self.model_device),
                                   reduction='mean')

        rmse_loss = self.rmse(pooler_output, labels.to(self.model_device), mask.to(self.model_device))
        rmse_loss_mean = self.rmse(pooler_output, labels.to(self.model_device), mask.to(self.model_device),
                                   reduction='mean')

        mae_loss = self.mae(pooler_output, labels.to(self.model_device), mask.to(self.model_device))
        mae_loss_mean = self.mae(pooler_output, labels.to(self.model_device), mask.to(self.model_device),
                                 reduction='mean')

        mdl_loss, mdl_f1 = self.mdl(pooler_output.detach(), labels.to(self.model_device), mask.to(self.model_device),
                                    return_f1=True)

        return {'test_loss': loss, 'test_loss_mean': loss_mean, 'test_rmse_loss': rmse_loss,
                'test_rmse_loss_mean': rmse_loss_mean, 'test_mae_loss': mae_loss, 'test_mae_loss_mean': mae_loss_mean,
                'test_mask_direction_loss': mdl_loss, 'test_mask_direction_f1': mdl_f1}

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

    def aggregate_metrics(self, outputs, stage='val'):
        avg_loss = torch.stack([x[stage + '_loss'] for x in outputs]).mean()
        avg_metrics = {}

        for metric in self.metrics:
            key_name = stage + '_' + metric
            try:
                avg_metrics.update(
                    {'avg_' + key_name: torch.stack(
                        [x[key_name] if isinstance(x[key_name], torch.Tensor)
                         else torch.tensor(x[key_name]) for x in outputs if x[key_name] >= 0.0]
                    ).mean()
                     })
            except RuntimeError:
                avg_metrics.update({'avg_' + key_name: torch.tensor(0.0)})

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
