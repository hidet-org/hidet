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


class NLPModel(BenchModel):
    def __init__(self, repo_name, model_name, label, batch_size: int, sequence_length: int):
        self.repo_name = repo_name
        self.model_name = model_name
        self.label = label
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __str__(self):
        return '{}/{}'.format(self.model_name, self.label)

    def model(self):
        import torch

        return torch.hub.load(self.repo_name, self.model_name, self.label)

    def example_inputs(self):
        import torch

        tokens_tensor = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.long, device='cuda')
        segments_tensors = torch.zeros((self.batch_size, self.sequence_length), dtype=torch.long, device='cuda')
        args = (tokens_tensor,)
        kwargs = {'token_type_ids': segments_tensors}
        return args, kwargs
