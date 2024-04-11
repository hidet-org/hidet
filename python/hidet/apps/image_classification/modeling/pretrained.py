from typing import Optional

import torch
from transformers import AutoModelForImageClassification, PretrainedConfig
from transformers import PreTrainedModel as TransformersPretrainedModel
from hidet.apps.modeling_outputs import ImageClassifierOutput
from hidet.apps.pretrained import PretrainedModel

import hidet


class PretrainedModelForImageClassification(PretrainedModel[ImageClassifierOutput]):
    @classmethod
    def create_pretrained_model(
        cls, config: PretrainedConfig, revision: Optional[str] = None, dtype: Optional[str] = None, device: str = "cuda"
    ):
        # dynamically load model subclass
        pretrained_model_class = cls.load_module(config.architectures[0])

        # load the pretrained huggingface model into cpu
        with torch.device("cuda"):  # reduce the time to load the model
            huggingface_token = hidet.option.get_option("auth_tokens.for_huggingface")
            torch_model: TransformersPretrainedModel = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name_or_path=config.name_or_path,
                torch_dtype=torch.float32,
                revision=revision,
                token=huggingface_token,
            )

        torch_model = torch_model.cpu()
        torch.cuda.empty_cache()

        dtype = cls.parse_dtype(config)
        hidet_model = pretrained_model_class(config)
        hidet_model.to(dtype=dtype, device=device)

        cls.copy_weights(torch_model, hidet_model)

        return hidet_model
