import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from encoding import IntervalEncoding, PositionalEncoding, IntervalEncodingTrainable

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
