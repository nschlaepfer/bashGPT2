import math
import mlx.core as mx
import mlx.nn as nn

from dataclasses import dataclass

@dataclass
class GPT2Config:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.split_size = config.n_embd
        self.scale = (self.split_size // self.n_head) ** -0.5

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def __call__(self, x, layer_past=None):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(3, axis=2)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = mx.concatenate([past_key, k], axis=-2)
            v = mx.concatenate([past_value, v], axis=-2)

        present = (k, v)

        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        att = mx.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y, present

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def __call__(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def __call__(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = [GPT2Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(self, input_ids, position_ids=None, past_key_values=None):
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = mx.arange(past_length, input_ids.shape[-1] + past_length, dtype=mx.int32)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        presents = []
        for block, layer_past in zip(self.h, past_key_values):
            hidden_states, present = block(hidden_states, layer_past=layer_past)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents

    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            outputs = self(input_ids)
            next_token_logits = outputs[0][:, -1, :] / temperature
            next_token = mx.random.categorical(next_token_logits, num_samples=1)
            input_ids = mx.concatenate([input_ids, next_token], axis=1)
            yield next_token

    def loss(self, input_ids, labels):
        logits, _ = self(input_ids)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        loss = nn.losses.cross_entropy(shift_logits, shift_labels)
        return mx.mean(loss)