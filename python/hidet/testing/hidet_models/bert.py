from typing import Optional
import math
import numpy as np
from hidet.graph import randn, zeros, array, ones
from hidet.graph import nn, Tensor
from hidet.graph import ops


# Acknowledgement: adopted the bert implementation from huggingface transformers package, with some simplification

class BertConfig:
    def __init__(self):
        self.vocab_size = 30522
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.max_position_embeddings = 512
        self.intermediate_size = 3072
        self.type_vocab_size = 2


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
            self,
            input_ids: Tensor,                          # [batch_size, seq_length] in [0, vocab_size)
            token_type_ids: Optional[Tensor] = None,    # [batch_size, seq_length] in [0, type_vocab_size)
            position_ids: Optional[Tensor] = None       # [batch_size, seq_length] in [0, max_position_embeddings)
    ):
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            ids = array(np.arange(seq_length).astype(np.int64))
            ids = ops.tile(ops.unsqueeze(ids, [0]), repeats=[batch_size, 1])
            position_ids = ids
        if token_type_ids is None:
            token_type_ids = zeros([batch_size, seq_length], dtype='int64')

        input_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeds + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('Multi head attention expects hidden_size % num_attention_heads == 0, '
                             'got {} and {}'.format(config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        batch_size, seq_length, hidden_size = x.shape
        x = x.reshape([batch_size, seq_length, self.num_attention_heads, self.attention_head_size])
        x = x.rearrange([[0, 2], [1], [3]])
        return x

    def forward(self, hidden_states: Tensor, attention_mask: Tensor):
        batch_size, seq_length, hidden_size = hidden_states.shape
        query = self.transpose_for_scores(self.query_layer(hidden_states))
        key = self.transpose_for_scores(self.key_layer(hidden_states))
        value = self.transpose_for_scores(self.value_layer(hidden_states))
        attention_scores = ops.batch_matmul(query, key.transpose([-1, -2])) / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = ops.softmax(attention_scores, axis=-1)
        context = ops.batch_matmul(attention_probs, value)
        context = context.reshape([batch_size, self.num_attention_heads, seq_length, self.attention_head_size])
        context = context.rearrange([[0], [2], [1, 3]])
        return context


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: Tensor, skip_hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + skip_hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        self.output_layer = BertSelfOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor):
        attention_output = self.self_attention(hidden_states, attention_mask)
        return self.output_layer(attention_output, hidden_states)


# well known as FeedForward
class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = nn.Gelu()

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: Tensor, skip_hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + skip_hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention_layer = BertAttention(config)
        self.intermediate_layer = BertIntermediate(config)
        self.output_layer = BertOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        attention_output = self.attention_layer(hidden_states, attention_mask)
        intermediate_output = self.intermediate_layer(attention_output)
        layer_output = self.output_layer(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        layers = []
        for _ in range(config.num_hidden_layers):
            layers.append(BertLayer(config))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor):
        for i, layer_module in enumerate(self.layers.submodules.values()):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def extend_attention_mask(self, attention_mask: Tensor) -> Tensor:
        # attention_mask: [batch_size, seq_length] in {0, 1}
        # return: [batch_size * num_attention_heads, 1, seq_length] in {-10000, 0}
        attention_mask = attention_mask.unsqueeze([1])
        attention_mask = ops.tile(attention_mask, repeats=[self.config.num_attention_heads, 1, 1])
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor):
        embeds = self.embeddings.forward(input_ids, token_type_ids)
        attention_mask = self.extend_attention_mask(attention_mask)
        hidden_states = self.encoder.forward(embeds, attention_mask)
        pooled_output = self.pooler(hidden_states)
        return [hidden_states, pooled_output]


def bert(batch_size=1, seq_length=128):
    config = BertConfig()
    model = BertModel(config)
    input_ids = array(np.random.randint(0, config.vocab_size, size=[batch_size, seq_length]))
    token_type_ids = zeros([batch_size, seq_length], dtype='int64')
    attention_mask = ones([batch_size, seq_length], dtype='int64')
    return model, [input_ids, token_type_ids, attention_mask]
