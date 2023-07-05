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
import pytest
from hidet.testing.models.llama import get_compiled_model, generate
from hidet.runtime.storage import current_memory_pool


# @pytest.mark.parametrize('device,opt', [('cuda', True)])
@pytest.mark.skip(reason='This test requires a lot of CPU memory > 32GB')
def test_llama(device, opt):
    model, config, tokenizer = get_compiled_model(device=device, opt=opt)

    text = generate('In the beginning was the Word.', model, tokenizer, config, num_tokens=12)
    print(text)
    expected = 'The Word was with God, and the Word was God.'
    assert text == expected

    text = generate(
        "A robot may not injure a human being or, through inaction", model, tokenizer, config, num_tokens=55
    )
    expected = (
        ', allow a human being to come to harm. A robot must obey orders given it by human beings'
        ' except where such orders would conflict with the First Law. A robot must protect its own'
        ' existence as long as such protection does not conflict with the First or Second Laws.'
    )

    print(text)
    assert text == expected

    print(current_memory_pool("cuda"))
    print(current_memory_pool("cpu"))
    print(current_memory_pool("vcuda"))

test_llama('cuda', True)
