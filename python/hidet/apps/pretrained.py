from typing import Generic, List, Set
import logging

import torch
from transformers import PretrainedConfig

from hidet.apps.registry import Registry
from hidet.graph import Tensor, nn
from hidet.graph.nn.module import R
from hidet.graph.tensor import from_torch
from hidet.utils import prod


logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class PretrainedModel(nn.Module[R], Registry, Generic[R]):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def copy_weights(cls, torch_model: torch.nn.Module, hidet_model: nn.Module):
        found_tensors: List[Tensor] = []
        for name, tensor in torch_model.state_dict().items():
            member = hidet_model
            for m_name in name.split("."):
                member = getattr(member, m_name)

            if not isinstance(member, Tensor):
                raise ValueError(
                    'PyTorch model "{}" defined a parameter "{}" that is not in the hidet model'.format(
                        torch_model.__class__.__name__, name
                    )
                )

            src = from_torch(tensor).to(member.dtype, member.device)

            if src.shape != member.shape:
                if prod(src.shape) == prod(member.shape):
                    logging.warning("Attempting to reshape parameter %s from %s to %s.", name, src.shape, member.shape)
                    src = src.reshape(member.shape)
                else:
                    raise ValueError(f"Parameter {name} shape mismatch, hidet: {member.shape}, torch: {src.shape}")

            found_tensors.append(member)
            member.copy_(src)

        buffer_names: Set[str] = set(name for name, _ in torch_model.named_buffers())
        for name, tensor in hidet_model.named_parameters():
            if tensor not in found_tensors and name not in buffer_names:
                raise ValueError(f"Parameter {name} in hidet model does not find equivalent in PyTorch model.")

    @classmethod
    def parse_dtype(cls, config: PretrainedConfig, default: str = "float16"):
        if config.torch_dtype:
            assert isinstance(config.torch_dtype, torch.dtype)
            return str(config.torch_dtype).rsplit(".", maxsplit=1)[-1]
        else:
            return default
