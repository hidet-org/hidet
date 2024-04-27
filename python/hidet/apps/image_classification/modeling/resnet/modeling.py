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

from dataclasses import asdict
from typing import Sequence
from transformers import ResNetConfig

from hidet.apps import PretrainedModel
from hidet.apps.image_classification.modeling.pretrained import PretrainedModelForImageClassification
from hidet.apps.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from hidet.apps.registry import RegistryEntry
from hidet.graph import nn
from hidet.graph.tensor import Tensor

PretrainedModel.register(
    arch="ResNetForImageClassification",
    entry=RegistryEntry(
        model_category="image_classification", module_name="resnet", klass="ResNetForImageClassification"
    ),
)

# Contents below reflects transformers.models.resnet.modeling_resnet.py
# with minor API changes


class ResNetConvLayer(nn.Module[Tensor]):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: bool = True
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.apply_activation = activation
        self.activation = nn.Relu()

    def forward(self, x: Tensor) -> Tensor:
        hidden_state = self.convolution(x)
        hidden_state = self.normalization(hidden_state)
        if self.apply_activation:
            hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetEmbeddings(nn.Module[Tensor]):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.embedder = ResNetConvLayer(config.num_channels, config.embedding_size, kernel_size=7, stride=2)
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_channels = config.num_channels

    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.embedder(pixel_values)
        embedding = self.pooler(embedding)
        return embedding


class ResNetShortCut(nn.Module[Tensor]):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        hidden_state = self.convolution(x)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class ResNetBottleNeckLayer(nn.Module[Tensor]):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 4):
        super().__init__()
        self.should_apply_shortcut = in_channels != out_channels or stride != 1
        if self.should_apply_shortcut:
            self.shortcut = ResNetShortCut(in_channels, out_channels, stride=stride)

        reduces_channels = out_channels // reduction
        layer = [
            ResNetConvLayer(in_channels, reduces_channels, kernel_size=1),
            ResNetConvLayer(reduces_channels, reduces_channels, stride=stride),
            ResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=False),
        ]

        self.layer = nn.Sequential(layer)
        self.activation = nn.Relu()

    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        if self.should_apply_shortcut:
            residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetStage(nn.Module[Tensor]):
    """
    A ResNet stage composed by stacked layers.
    """

    def __init__(self, config: ResNetConfig, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2):
        super().__init__()

        if config.layer_type != "bottleneck":
            raise NotImplementedError(
                "Only ResNet bottleneck layers supported. See ResNetBasicLayer in transformers source."
            )

        if config.hidden_act != "relu":
            raise NotImplementedError("Only ReLU supported for ResNet activation.")

        self.layers = nn.Sequential(
            # downsampling is done in the first layer with stride of 2
            ResNetBottleNeckLayer(in_channels, out_channels, stride=stride),
            *[ResNetBottleNeckLayer(out_channels, out_channels) for _ in range(depth - 1)],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers.forward(x)


class ResNetEncoder(nn.Module[BaseModelOutput]):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        stages = [
            ResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        ]

        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            stages.append(ResNetStage(config, in_channels, out_channels, depth=depth))

        self.stages: nn.ModuleList = nn.ModuleList(stages)

    def forward(self, hidden_state: Tensor) -> BaseModelOutput:
        hidden_states = [hidden_state]

        for stage_module in self.stages:
            if stage_module is not None:
                hidden_state = stage_module(hidden_state)
                hidden_states.append(hidden_state)

        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=hidden_states)


class ResNetClassifier(nn.Sequential):
    class Flatten(nn.Module):
        def __init__(self, dims: Sequence[int]):
            super().__init__()
            self.dims = dims

        def forward(self, x: Tensor) -> Tensor:
            return x.squeeze(self.dims)

    def __init__(self, config: ResNetConfig):
        super().__init__()
        assert config.num_labels > 0

        layers = [self.Flatten((2, 3)), nn.Linear(config.hidden_sizes[-1], config.num_labels)]
        for idx, module in enumerate(layers):
            self.__setattr__(str(idx), module)


class ResNetModel(nn.Module[BaseModelOutputWithPooling]):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.config = config
        self.embedder = ResNetEmbeddings(config)
        self.encoder = ResNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input_images: Tensor) -> BaseModelOutputWithPooling:
        embedding_output = self.embedder(input_images)
        encoder_outputs: BaseModelOutput = self.encoder(embedding_output)

        pooled_output = self.pooler(encoder_outputs.last_hidden_state)

        return BaseModelOutputWithPooling(**asdict(encoder_outputs), pooler_output=pooled_output)


class ResNetForImageClassification(PretrainedModelForImageClassification):
    def __init__(self, config: ResNetConfig):
        assert isinstance(config, ResNetConfig)
        super().__init__(config)
        self.num_labels = config.num_labels
        self.resnet = ResNetModel(config)
        # classification head
        self.classifier = ResNetClassifier(config)

    def forward(self, input_images: Tensor) -> ImageClassifierOutput:
        outputs: BaseModelOutputWithPooling = self.resnet(input_images)

        logits = self.classifier(outputs.pooler_output)

        return ImageClassifierOutput(**asdict(outputs), logits=logits)
