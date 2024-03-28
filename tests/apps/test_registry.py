import pytest
from hidet.apps import Registry, hf
from hidet.apps.image_classification.modeling.resnet.modeling import ResNetForImageClassification
from transformers import PretrainedConfig


@pytest.mark.slow
@pytest.mark.parametrize('model_name', ["microsoft/resnet-50"])
def test_load_module(model_name: str):
    config: PretrainedConfig = hf.load_pretrained_config(model_name)
    assert Registry.load_module(config) is ResNetForImageClassification


if __name__ == '__main__':
    pytest.main([__file__])
