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
import torch
import torch.backends.cudnn
import pytest
import hidet
from hidet.testing import device_to_torch


@pytest.mark.skip(reason="The repo seems invalid and the model is not available now.")
@pytest.mark.parametrize('batch_size', [1])
@pytest.mark.parametrize('seq_length', [128])
@pytest.mark.parametrize('use_tensor_core', [False, True])
@pytest.mark.parametrize('dynamic', [False])  # TODO: enable dynamic when torch dynamo is fixed
def test_bert(batch_size: int, seq_length: int, use_tensor_core, dynamic, device):
    torch_device = device_to_torch(device)
    tokens_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long, device=torch_device)
    segments_tensors = torch.zeros((batch_size, seq_length), dtype=torch.long, device=torch_device)
    args = (tokens_tensor.to(torch_device),)
    kwargs = {'token_type_ids': segments_tensors.to(torch_device)}
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased').to(torch_device).eval()
    model_opt = torch.compile(model, backend='hidet', mode=None, dynamic=dynamic)
    y1 = model(*args, **kwargs).last_hidden_state

    try:
        hidet.torch.dynamo_config.use_tensor_core(use_tensor_core)
        y2 = model_opt(*args, **kwargs).last_hidden_state
        tol = 2e-2
        torch.testing.assert_close(y1, y2, atol=tol, rtol=tol)
    finally:
        # in case of failure, reset the config
        hidet.torch.dynamo_config.reset()


if __name__ == '__main__':
    pytest.main([__file__])
