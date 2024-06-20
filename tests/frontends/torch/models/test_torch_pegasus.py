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
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


@pytest.mark.slow
def test_pegasus():
    src_text = [
        "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest "
        "structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, "
        "the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a "
        "title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the "
        "first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top "
        "of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding "
        "transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau "
        "Viaduct. "
    ]

    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail").cuda().eval()

    model.model.encoder = torch.compile(model.model.encoder, backend='hidet', mode=None, dynamic=True)
    model.model.decoder = torch.compile(model.model.decoder, backend='hidet', mode=None, dynamic=True)

    batch = tokenizer(src_text, truncation=True, return_tensors="pt").to('cuda')
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
