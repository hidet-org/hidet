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
from hidet.apps.image_classification.pipeline.pipeline import ImageClassificationPipeline
from hidet.apps.image_classification.processing.image_processor import ChannelDimension
import pytest
from datasets import load_dataset


def test_image_classifier_pipeline():
    dataset = load_dataset("huggingface/cats-image", split="test", trust_remote_code=True)

    pipeline = ImageClassificationPipeline("microsoft/resnet-50", batch_size=1, kernel_search_space=0)

    res = pipeline(dataset["image"], input_data_format=ChannelDimension.CHANNEL_LAST, top_k=3)

    assert len(res) == 1
    assert all([len(x) == 3 for x in res])


if __name__ == "__main__":
    pytest.main([__file__])
