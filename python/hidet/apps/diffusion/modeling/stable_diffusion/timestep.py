import math
from typing import Callable, Optional
import hidet
from hidet import nn
from hidet.graph.tensor import Tensor
from hidet.graph.ops import exp, expand_dims, concat, sin, cos, pad


class TimestepEmbedding(nn.Module[Tensor]):
    def __init__(
        self, in_channels: int, time_embed_dim: int, act_fn: Callable[[Tensor], Tensor], out_dim: Optional[int] = None
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = act_fn

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

    def forward(self, sample) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module[Tensor]):
    def __init__(self, num_channels: int, flip_sin_to_cos: int, downscale_freq_shift: float = 0):
        super().__init__()

        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    # TODO: substantially the same as LlamaRotaryEmbedding, except timesteps can be any arithmetic sequence
    # not necessarily arange with common difference 1. So caching not effective.
    def forward(self, timesteps: Tensor) -> Tensor:
        MAX_PERIOD = 10_000

        assert len(timesteps.shape) == 1

        half_dim = self.num_channels // 2
        exponent = -math.log(MAX_PERIOD) * hidet.arange(0, stop=half_dim, dtype="float32", device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = exp(exponent)
        emb = expand_dims(timesteps, axis=1) * expand_dims(emb, axis=0)

        if self.flip_sin_to_cos:
            emb = concat([cos(emb), sin(emb)], axis=1)
        else:
            emb = concat([sin(emb), cos(emb)], axis=1)

        if self.num_channels % 2 == 1:
            emb = pad(emb, [0, 1, 0, 0])

        return emb
