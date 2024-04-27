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
from hidet.apps import Registry, hf
from hidet.apps.image_classification.modeling.resnet.modeling import ResNetForImageClassification
from transformers import PretrainedConfig


@pytest.mark.slow
@pytest.mark.parametrize('model_name', ["microsoft/resnet-50"])
def test_load_module(model_name: str):
    config: PretrainedConfig = hf.load_pretrained_config(model_name)
    assert Registry.load_module(config.architectures[0]) is ResNetForImageClassification


if __name__ == '__main__':
    pytest.main([__file__])
