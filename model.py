import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from torch.utils.data import DataLoader

from data import DataProcessor
from encoding import IntervalEncoding, PositionalEncoding, IntervalEncodingTrainable
from losses import MaskedMSELoss, MaskedRMSELoss, MaskedMAELoss, MaskedDirectionLoss

positional_encoders = {'interval': IntervalEncoding, 'positional': PositionalEncoding,
                       'interval-trainable': IntervalEncodingTrainable}


class Embeddings(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.pos_encoder = positional_encoders.get(configs.get('encoder', 'interval'), IntervalEncoding)(configs)
        # IntervalEncoding(configs) if configs.encoder == 'interval' else PositionalEncoding(configs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(input_ids)

        return x


class FNetBlock(nn.Module):

    def __init__(self, configs):
        super().__init__()

        # self.fft = FNetBasicFourierTransform(configs)

    def forward(self, x, attention_mask=None):
        # x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = torch.fft.fft2(x).real
        # x = self.fft(x)

        return x, attention_mask


class ModelFeedForward(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.linear1 = nn.Linear(configs.hidden_size, configs.intermediate_size)
        self.activation = nn.GELU() if configs.hidden_act == 'gelu' else nn.ReLU()
        # self.dropout1 = nn.Dropout(configs.intermediate_dropout_prob)
        self.linear2 = nn.Linear(configs.intermediate_size, configs.hidden_size)
        self.dropout2 = nn.Dropout(configs.intermediate_dropout_prob)

        self.configs = configs

        # self.ff = nn.Sequential(
        # nn.Linear(configs.input_size, configs.hidden_size),
        # nn.GELU(),
        # nn.Dropout(configs.feed_forward_dropout_prob),
        # nn.Linear(configs.hidden_size, configs.input_size),
        # nn.Dropout(configs.feed_forward_dropout_prob)
        # )

    def forward(self, x):
        hidden_states = self.linear1(x)
        hidden_states = self.activation(hidden_states)
        output_states = self.linear2(hidden_states)
        # drop_output = self.dropout1(output)

        # return self.dropout2(self.linear2(drop_output))
        return self.dropout2(output_states)

    def init_weights(self, init_range):
        self.linear1.weight.data.uniform_(-init_range, init_range)
        self.linear2.weight.data.uniform_(-init_range, init_range)


class ModelLayer(nn.Module):

    def __init__(self, configs, bert_self_attn=False):
        super().__init__()

        self.attn = None
        self.intermediate = None
        self.output = None

        if bert_self_attn:
            self.attn = BertAttention(configs)
            self.intermediate = BertIntermediate(configs)
            self.output = BertOutput(configs)
        else:
            self.attn = FNetBlock(configs)

        self.ff = ModelFeedForward(configs)
        self.attn_norm = nn.LayerNorm(configs.hidden_size, eps=configs.layer_norm_eps)
        self.ff_norm = nn.LayerNorm(configs.hidden_size, eps=configs.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        # x = self.attn(self.attn_norm(x)) + x
        # return self.ff(self.ff_norm(x)) + x

        attn_outputs = self.attn(hidden_states, attention_mask)
        attn_output = attn_outputs[0]

        # print(type(self.attn))

        # if isinstance(self.attn, BertAttention):
        #     print(hidden_states[0].tolist())

        # attention_mask, x = attn_output[:2]

        if isinstance(self.attn, BertAttention):
            intermediate_output = self.intermediate(attn_output)
            return self.output(intermediate_output, attn_output), attention_mask

        hidden_states = self.attn_norm(attn_output + hidden_states)

        return self.ff_norm(self.ff(hidden_states) + hidden_states), attention_mask

    def init_weights(self, init_range):
        # initrange = self.config.initializer_range #0.1
        # self.attn.weight.data.uniform_(-init_range, init_range)

        self.ff.init_weights(init_range)

        if isinstance(self.attn, BertAttention):
            self.attn.self.query.weight.data.uniform_(-init_range, init_range)  # #self.attn.init_weights(init_range)
            self.attn.self.key.weight.data.uniform_(-init_range, init_range)
            self.attn.self.value.weight.data.uniform_(-init_range, init_range)


class FNetEncoder(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(configs.num_layers):
            self.layers.append(ModelLayer(configs, bert_self_attn=i in configs.spec_attention_layers))

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x, attn = layer(x, attention_mask)

        return x

    def init_weights(self, init_range):
        for layer in self.layers:
            layer.init_weights(init_range)


class ModelPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 4)
        # self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(hidden_states)
        # pooled_output = self.activation(pooled_output)
        return pooled_output


class Model(nn.Module):

    def __init__(self, configs, device):
        super().__init__()

        self.device = device

        self.embeddings = Embeddings(configs)
        self.encoder = FNetEncoder(configs)
        self.pooler = ModelPooler(configs)

        self.init_weights(configs.initializer_range)

    # TODO: rewrite with torch.nn.Module.apply(fn)
    def init_weights(self, init_range):
        self.apply(lambda m: initialize_weights(m, init_range))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None):
        embeddings = self.embeddings(input_ids)

        input_shape = input_ids.size()

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        sequence_output = self.encoder(embeddings, extended_attention_mask)
        pooler_output = self.pooler(sequence_output)

        return sequence_output, pooler_output

    def get_extended_attention_mask(self, attention_mask, input_shape):

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(self.device)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


@torch.no_grad()
def initialize_weights(m, init_range):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-init_range, init_range)


class ModelForFM(pl.LightningModule):

    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = Model(configs, self.model_device)

        self.criterion = MaskedMSELoss(reduction=configs.get('reduction_strategy', 'sum'))
        self.rmse = MaskedRMSELoss(reduction=configs.get('reduction_strategy', 'sum'))
        self.mae = MaskedMAELoss(reduction=configs.get('reduction_strategy', 'sum'))
        self.mdl = MaskedDirectionLoss(reduction=configs.get('reduction_strategy', 'sum'))

        self.data_processor = DataProcessor(configs)
        self.metrics = configs.get('metrics', [])

        if configs.get('weights_path', None) is not None:
            self.load_weights(configs['weights_path'])

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

    def load_dataset(self, dataset_path: str = None) -> tuple:
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

    def load_weights(self, weights_path):
        try:
            if weights_path.endswith('.ckpt'):
                self.load_state_dict(torch.load(weights_path)['state_dict'])
            else:
                self.load_state_dict(torch.load(weights_path))

        except Exception as e:
            print(e)
            print('Failed to load weights from {}'.format(weights_path))
