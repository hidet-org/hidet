from typing import Optional
from hidet.graph import nn
from hidet.graph.tensor import Tensor


class Downsample2D(nn.Module):
    def __init__(self, channels: int, output_channels: Optional[int], downsample_padding: int):
        super().__init__()

        self.channels = channels
        self.out_channels = output_channels or channels

        self.conv = nn.Conv2d(
            self.channels, self.out_channels, kernel_size=3, stride=2, padding=downsample_padding, bias=True
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        assert hidden_states.shape[1] == self.channels

        return self.conv(hidden_states)
