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
from hidet.apps.image_classification.processing.image_processor import ChannelDimension
from hidet.apps.image_classification.processing.resnet.processing import ResNetImageProcessor
import pytest
import torch


def test_resnet_processor_resize():
    # Channel first
    image = torch.zeros((3, 10, 15), dtype=torch.uint8)
    image += torch.arange(1, 16)

    processor = ResNetImageProcessor(size=4, do_rescale=False, do_normalize=False)
    res = processor(image, input_data_format=ChannelDimension.CHANNEL_FIRST)
    assert res.shape == (1, 3, 4, 4)
    assert ((0 < res.torch()) & (res.torch() < 15)).all()

    # Channel last
    image = torch.zeros((10, 15, 3), dtype=torch.uint8)
    image += torch.arange(1, 16).view(1, 15, 1)

    processor = ResNetImageProcessor(size=4, do_rescale=False, do_normalize=False)
    res = processor(image, input_data_format=ChannelDimension.CHANNEL_LAST)
    assert res.shape == (1, 3, 4, 4)
    assert ((0 < res.torch()) & (res.torch() < 15)).all()

    # Batch resize
    images = []

    import random

    random.seed(0)

    for _ in range(10):
        rows = random.randint(10, 20)
        cols = random.randint(10, 20)
        tensor = torch.randint(1, 9, (rows, cols, 3), dtype=torch.uint8)
        images.append(tensor)

    res = processor(images, input_data_format=ChannelDimension.CHANNEL_LAST)
    assert res.shape == (10, 3, 4, 4)
    assert ((0 <= res.torch()) & (res.torch() < 10)).all()


if __name__ == "__main__":
    pytest.main([__file__])
