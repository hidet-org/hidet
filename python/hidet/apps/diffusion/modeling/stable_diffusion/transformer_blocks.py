from typing import Optional
from hidet.graph import nn, ops
from hidet.graph.tensor import Tensor


class FeedForward(nn.Module[Tensor]):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        activation_fn: str = "geglu",
        inner_dim: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn != "geglu":
            raise NotImplementedError("Expected geglu for feedforward activation.")

        act_fn = nn.Geglu(dim, inner_dim, bias=bias)

        self.net = []
        self.net.append(act_fn)
        self.net.append(nn.Identity())  # replaces dropout
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        self.net = nn.Sequential(self.net)

    def forward(self, x) -> Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module[Tensor]):
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = nn.CrossAttention(
            dim, heads=kwargs["num_attention_heads"], dim_head=kwargs["attention_head_dim"], upcast=True, out_bias=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = nn.CrossAttention(
            dim,
            cross_attention_dim=kwargs["cross_attention_dim"],
            heads=kwargs["num_attention_heads"],
            dim_head=kwargs["attention_head_dim"],
            upcast=True,
            out_bias=True,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, activation_fn="geglu", bias=True)

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor, temperature_scaling: float = 1.0) -> Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(norm_hidden_states, temperature_scaling=temperature_scaling)

        hidden_states = attn_output + hidden_states
        if len(hidden_states.shape) == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)

        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)

        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if len(hidden_states.shape) == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Transformer2DModel(nn.Module[Tensor]):
    def __init__(self, **kwargs):
        super().__init__()

        inner_dim = kwargs["num_attention_heads"] * kwargs["attention_head_dim"]
        self.use_linear_projection = kwargs["use_linear_projection"]

        self.norm = nn.GroupNorm(kwargs["resnet_groups"], kwargs["input_channels"], eps=1e-6, affine=True)
        if kwargs["use_linear_projection"]:
            self.proj_in = nn.Linear(kwargs["input_channels"], inner_dim)
        else:
            self.proj_in = nn.Conv2d(kwargs["input_channels"], inner_dim, kernel_size=1)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, **kwargs) for _ in range(kwargs["num_layers"])]
        )

        self.output_channels = (
            kwargs["input_channels"] if kwargs.get("output_channels", None) is None else kwargs["output_channels"]
        )

        if kwargs["use_linear_projection"]:
            self.proj_out = nn.Linear(inner_dim, kwargs["input_channels"])
        else:
            self.proj_out = nn.Conv2d(inner_dim, kwargs["input_channels"], kernel_size=1)

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor, temperature_scaling: float = 1.0) -> Tensor:
        bs, _, h, w = hidden_states.shape
        residuals = hidden_states
        hidden_states = self.norm(hidden_states)

        def compress_hidden_states(x):
            return ops.permute_dims(x, (0, 2, 3, 1)).reshape((bs, h * w, x.shape[1]))

        def decompress_hidden_states(x):
            return ops.permute_dims(x.reshape((bs, h, w, inner_dim)), (0, 3, 1, 2)).contiguous()

        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = compress_hidden_states(hidden_states)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = compress_hidden_states(hidden_states)
            hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, temperature_scaling=temperature_scaling)

        if not self.use_linear_projection:
            hidden_states = decompress_hidden_states(hidden_states)
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = decompress_hidden_states(hidden_states)

        return hidden_states + residuals
