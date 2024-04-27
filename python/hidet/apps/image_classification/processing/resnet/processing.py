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
from typing import List, Optional, Union
import numpy as np
import torch
import PIL

from hidet.graph.flow_graph import FlowGraph, trace_from
from hidet.graph.ops.image import resize2d
from hidet.graph.tensor import Tensor, from_torch, symbol

import hidet
from ..image_processor import BaseImageProcessor, ChannelDimension, ImageInput


class ResNetImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,
        size: int = 224,
        crop_pct: float = 0.875,
        resample: str = "cubic",
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        dtype: str = "float32",
        device: str = "cuda",
        **kwargs,
    ) -> None:
        """
        Pre-process images before ResNet model. Produces square images of (size, size) and transforms
        input images to channel first.
        Assumes inputs are uint8 RGB images on CPU memory.
        Default values taken from `AutoImageProcessor.from_pretrained("microsoft/resnet-50")`, not ImageNet
        standards.
        See transformers library ConvNextImageProcessor for reference implementation.
        """
        super().__init__(dtype, device)

        assert 0 < crop_pct < 1

        self.do_resize = do_resize
        self.size = size
        self.crop_pct = crop_pct
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]

        resize_inputs: Tensor = symbol([1, 3, "h", "w"], dtype="uint8", device="cuda")
        resize_outputs = self.resize(resize_inputs.to(self.dtype))
        resize_graph: FlowGraph = trace_from(resize_outputs, resize_inputs)

        scaling_inputs: Tensor = symbol(["n", 3, "h", "w"], dtype="int32" if self.do_resize else "uint8", device="cuda")
        scaling_outputs = scaling_inputs.to(self.dtype)
        if do_rescale:
            scaling_outputs = self.rescale(scaling_outputs, scale=rescale_factor)
        if do_normalize:
            scaling_outputs = self.normalize(scaling_outputs, self.image_mean, self.image_std)
        rescale_graph: FlowGraph = trace_from(scaling_outputs, scaling_inputs)

        resize_graph = hidet.graph.optimize(resize_graph)
        rescale_graph = hidet.graph.optimize(rescale_graph)

        self.resize_graph = resize_graph.build(space=2)
        self.rescale_graph = rescale_graph.build(space=2)

    def preprocess(
        self, images: ImageInput, input_data_format: ChannelDimension, _do_resize: Optional[bool] = None, **kwargs
    ) -> Tensor:
        assert isinstance(images, (PIL.Image.Image, np.ndarray, torch.Tensor, list))

        do_resize = _do_resize if _do_resize is not None else self.do_resize

        def _preprocess_one(images):
            if isinstance(images, PIL.Image.Image):
                images = np.asarray(images)
                # fall through
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)

            assert isinstance(images, torch.Tensor)

            if len(images.shape) == 2:
                # broadcast grayscale to 3 channels
                images = images.expand(3, -1, -1)
            if len(images.shape) == 3:
                # batch size 1
                images = images.reshape(1, *images.shape)

            if input_data_format == ChannelDimension.CHANNEL_LAST:
                # change to channel first
                images = images.permute(0, 3, 1, 2)

            hidet_images: Tensor = from_torch(images.contiguous()).to(device=self.device)

            if do_resize:
                hidet_images = self.resize_graph.run_async([hidet_images])[0]

            if self.do_rescale or self.do_normalize:
                hidet_images = self.rescale_graph.run_async([hidet_images])[0]
            return hidet_images

        def _preprocess_many(images):
            common_type = type(images[0])
            assert all(isinstance(image, common_type) for image in images)

            if not do_resize:
                # batching images requires same size
                common_size = images[0].shape
                assert all(image.shape == common_size for image in images)

            # change to torch
            if issubclass(common_type, PIL.Image.Image):
                images = [torch.from_numpy(np.asarray(image).copy()) for image in images]
            elif common_type is np.ndarray:
                images = [torch.from_numpy(image) for image in images]

            if input_data_format == ChannelDimension.CHANNEL_FIRST:
                images = [image.expand(3, -1, -1) if len(image.shape) < 3 else image for image in images]
            else:
                images = [image.unsqueeze(-1).repeat(1, 1, 3) if len(image.shape) < 3 else image for image in images]

            if input_data_format == ChannelDimension.CHANNEL_LAST:
                images = [image.permute(2, 0, 1).contiguous() for image in images]

            if do_resize:
                resized_images = []
                for image in images:
                    image = from_torch(image.reshape(1, *image.shape)).to(device=self.device)
                    image = self.resize_graph.run_async([image])[0]
                    image = image.torch()
                    resized_images.append(image)

                images = resized_images

            # combine to single tensor, recurse
            images = torch.stack(images).squeeze()
            return self.preprocess(images, ChannelDimension.CHANNEL_FIRST, _do_resize=False)

        return _preprocess_many(images) if isinstance(images, list) else _preprocess_one(images)

    def resize(self, image: Tensor):
        """
        If size is <384, resize to size / crop_pct and then apply center crop (to preserve image quality).
        Mirrors ConvNextImageProcessor resize operation. Assumes input image with shape (bs, 3, h, w).
        """
        assert len(image.shape) == 4

        if self.size < 384:
            resize_shortest_edge = int(self.size / self.crop_pct)
            image = resize2d(image, size=(resize_shortest_edge, resize_shortest_edge), method='cubic').to(dtype="int32")

            x = self.center_square_crop(image, self.size)
            return x

        else:
            return resize2d(image, size=(self.size, self.size), method=self.resample)


BaseImageProcessor.register(arch="ResNetForImageClassification", processor_class=ResNetImageProcessor)
