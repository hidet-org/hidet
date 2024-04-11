from typing import Optional

import torch
from transformers import AutoConfig, PretrainedConfig
from diffusers import StableDiffusionPipeline

import hidet


def _get_hf_auth_token():
    return hidet.option.get_option('auth_tokens.for_huggingface')


def load_pretrained_config(model: str, revision: Optional[str] = None) -> PretrainedConfig:
    huggingface_token = _get_hf_auth_token()
    return AutoConfig.from_pretrained(model, revision=revision, token=huggingface_token)


def load_diffusion_pipeline(name: str, revision: Optional[str] = None, device: str = "cuda") -> StableDiffusionPipeline:
    huggingface_token = _get_hf_auth_token()
    with torch.device(device):
        return StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=name, torch_dtype=torch.float32, revision=revision, token=huggingface_token
        )
