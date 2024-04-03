from typing import Optional
from hidet.graph import nn
from hidet.graph.tensor import Tensor
from hidet.graph.ops import resize2d


class Upsample2D(nn.Module):
    def __init__(self, channels: int, **kwargs):
        super().__init__()

        self.channels = channels
        self.out_channels = kwargs.get("output_channels", None) or channels

        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, hidden_states: Tensor, output_size: Optional[int] = None):
        assert hidden_states.shape[1] == self.channels

        if output_size is None:
            hidden_states = resize2d(hidden_states, scale_factor=2.0, method="nearest")
        else:
            hidden_states = resize2d(hidden_states, size=output_size, method="nearest")

        return self.conv(hidden_states)
