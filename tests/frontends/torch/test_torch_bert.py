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


@pytest.mark.parametrize('batch_size', [1])
@pytest.mark.parametrize('seq_length', [128])
@pytest.mark.parametrize('use_fp16,use_tensor_core', [(False, False), (False, True), (True, True)])
def test_bert(batch_size: int, seq_length: int, use_fp16, use_tensor_core):
    tokens_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long, device='cuda')
    segments_tensors = torch.zeros((batch_size, seq_length), dtype=torch.long, device='cuda')
    args = (tokens_tensor.cuda(),)
    kwargs = {'token_type_ids': segments_tensors.cuda()}
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased').cuda().eval()
    model_opt = torch.compile(model, backend='hidet')
    y1 = model(*args, **kwargs).last_hidden_state

    hidet.torch.dynamo_config.use_fp16(use_fp16)
    hidet.torch.dynamo_config.use_tensor_core(use_tensor_core)

    y2 = model_opt(*args, **kwargs).last_hidden_state
    torch.testing.assert_close(y1, y2, atol=1e-2, rtol=1e-2)

    hidet.torch.dynamo_config.reset()


if __name__ == '__main__':
    pytest.main([__file__])
