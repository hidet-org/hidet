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
from hidet.cli.bench.model import BenchModel


class VisionModel(BenchModel):
    def __init__(self, model_name: str, batch_size, channels: int, height: int, width: int):
        self.model_name = model_name
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width

    def __str__(self):
        return self.model_name

    def model(self):
        import torch

        return torch.hub.load('pytorch/vision:v0.6.0', self.model_name, pretrained=True, verbose=False)

    def example_inputs(self):
        import torch

        args = (torch.randn(self.batch_size, self.channels, self.height, self.width),)
        kwargs = {}
        return args, kwargs
