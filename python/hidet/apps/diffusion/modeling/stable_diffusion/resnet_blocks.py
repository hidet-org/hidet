from typing import Callable, Optional
from hidet.graph import nn
from hidet.graph.tensor import Tensor
from hidet.graph.ops import split


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: Optional[int],
        resnet_groups: int,
        temb_channels: int,
        resnet_time_scale_shift: str,
        resnet_act_fn: Callable,
        resnet_eps: float = 1e-05,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        output_channels = output_channels or input_channels
        groups_out = resnet_groups

        self.norm1 = nn.GroupNorm(num_groups=resnet_groups, num_channels=input_channels, eps=resnet_eps)

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, bias=True
        )

        self.time_embedding_norm = resnet_time_scale_shift

        self.time_emb_proj = None
        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, output_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * output_channels)
            else:
                raise ValueError(f"Unknown time_embedding_norm: {self.time_embedding_norm}")

        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=output_channels, eps=resnet_eps)

        if dropout != 0.0:
            raise NotImplementedError("No dropout should be used for inference")

        self.conv2 = nn.Conv2d(
            in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, bias=True
        )

        self.nonlinearity = resnet_act_fn

        self.use_in_shortcut = input_channels != output_channels
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=1, bias=True
            )

    def forward(self, x: Tensor, temb: Tensor):
        input_tensor = x
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            x = x + temb
            x = self.norm2(x)
        elif self.time_embedding_norm == "scale_shift":
            time_scale, time_shift = split(temb, 2, axis=1)[:2]
            x = self.norm2(x)
            x = x * (1 + time_scale) + time_shift
        else:
            x = self.norm2(x)

        x = self.nonlinearity(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + x
        return output_tensor
