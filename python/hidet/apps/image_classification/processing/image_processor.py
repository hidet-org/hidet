# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import Enum, auto, unique
from typing import Dict, List, Optional, Sequence, Type, Union
import torch
import numpy as np
from hidet.graph.tensor import Tensor, from_torch


@unique
class ChannelDimension(Enum):
    CHANNEL_FIRST = auto()
    CHANNEL_LAST = auto()
    CHANNEL_SINGLE = auto()


ImageInput = Union[
    "PIL.Image.Image", np.ndarray, "torch.Tensor", List["PIL.Image.Image"], List[np.ndarray], List["torch.Tensor"]
]  # noqa


class BaseImageProcessor:

    processor_registry: Dict[str, Type["BaseImageProcessor"]] = {}

    def __init__(self, dtype: Optional[str] = None, device: str = "cuda"):
        super().__init__()

        self.dtype = dtype
        self.device = device

    @classmethod
    def register(cls, arch: str, processor_class: Type["BaseImageProcessor"]):
        cls.processor_registry[arch] = processor_class

    @classmethod
    def load_module(cls, arch: str):
        return cls.processor_registry[arch]

    def __call__(self, images: ImageInput, **kwargs) -> Tensor:
        return self.preprocess(images, **kwargs)

    def preprocess(self, images: ImageInput, **kwargs) -> Tensor:
        raise NotImplementedError("Image processors should implement their own preprocess step.")

    def rescale(self, image: Tensor, scale: float) -> Tensor:
        return image * scale

    def normalize(
        self, image: Tensor, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]]
    ) -> Tensor:
        """
        Normalize image on per channel basis as
        (mean - pixel) / std
        mean and std are broadcast across channels if scalar value provided.
        """
        num_channels: int = image.shape[-3]

        if isinstance(mean, Sequence):
            if len(mean) != num_channels:
                raise ValueError(f"means need {num_channels} values, one for each channel, got {len(mean)}.")
        else:
            mean = [mean] * num_channels
        channel_means = from_torch(torch.Tensor(mean).view(num_channels, 1, 1)).to(self.dtype, self.device)

        if isinstance(std, Sequence):
            if len(std) != num_channels:
                raise ValueError(f"stds need {num_channels} values, one for each channel, got {len(std)}.")
        else:
            std = [std] * num_channels
        channel_stds = from_torch(torch.Tensor(std).view(num_channels, 1, 1)).to(self.dtype, self.device)

        return (image - channel_means) / channel_stds

    def center_square_crop(self, image: Tensor, size: int):
        assert image.shape[-2:] >= (size, size)

        pad_width = image.shape[-2] - size
        start = (pad_width // 2) + (pad_width % 2)
        end = image.shape[-2] - (pad_width // 2)

        return image[:, :, start:end, start:end]
