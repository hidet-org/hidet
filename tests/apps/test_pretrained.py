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
from hidet.apps import PretrainedModel, hf
from hidet.apps.image_classification.modeling.resnet.modeling import ResNetForImageClassification
from hidet.option import get_option
from transformers import AutoModelForImageClassification, PretrainedConfig, ResNetConfig


@pytest.mark.parametrize(
    "model_name, dtype",
    [
        ("microsoft/codebert-base", "float16"),  # resolve to default float16
        ("microsoft/resnet-50", "float32"),  # use config float32
    ],
)
def test_parse_dtype(model_name: str, dtype: str):
    config: PretrainedConfig = hf.load_pretrained_config(model_name)
    assert PretrainedModel.parse_dtype(config) == dtype


def test_copy_weights():

    with torch.device("cuda"):
        config: ResNetConfig = hf.load_pretrained_config("microsoft/resnet-50")
        huggingface_token = get_option("auth_tokens.for_huggingface")

        torch_model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=config.name_or_path, torch_dtype=torch.float32, token=huggingface_token
        )
        hidet_model = ResNetForImageClassification(config)
        hidet_model.to(dtype="float32", device="cuda")
        PretrainedModel.copy_weights(torch_model, hidet_model)

        normalization_stage = (
            hidet_model.resnet.encoder.stages._submodules["0"]
            .layers._submodules["0"]
            .layer._submodules["0"]
            .normalization
        )
        weight_set = [
            normalization_stage.weight,
            normalization_stage.bias,
            normalization_stage.running_mean,
            normalization_stage.running_var,
            hidet_model.classifier._submodules["1"].weight,
            hidet_model.resnet.embedder.embedder.convolution.weight,
        ]

        for weight in weight_set:
            weight = weight.torch()
            assert not torch.equal(weight, torch.zeros_like(weight))


if __name__ == "__main__":
    pytest.main([__file__])
