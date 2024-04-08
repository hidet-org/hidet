from typing import Optional, Set, List, Type
from transformers import AutoModelForCausalLM
import torch
import hidet.option
from hidet.ir.type import DataType
from hidet.graph.tensor import Tensor
from hidet.apps.llm import nn
from hidet.apps.llm.nn.attention import AttentionState


def copy_weights(torch_model: torch.nn.Module, hidet_model: nn.Module):
    import hidet  # pylint: disable=redefined-outer-name

    found_tensors: List[Tensor] = []
    for name, tensor in torch_model.named_parameters():
        member = hidet_model
        for m_name in name.split('.'):
            member = getattr(member, m_name)

        if not isinstance(member, hidet.Tensor):
            raise ValueError(
                'PyTorch model "{}" defined a parameter "{}" that is not in the hidet model'.format(
                    torch_model.__class__.__name__, name
                )
            )

        src = hidet.from_torch(tensor).to(member.dtype, member.device)
        if len(src.shape) != len(member.shape) or any(a != b for a, b in zip(src.shape, member.shape)):
            raise ValueError(f"Parameter {name} shape mismatch, hidet: {member.shape}, torch: {src.shape}")
        found_tensors.append(member)
        member.copy_(src)

    buffer_names: Set[str] = set(name for name, _ in torch_model.named_buffers())

    for name, tensor in hidet_model.named_parameters():
        if tensor not in found_tensors and name not in buffer_names:
            raise ValueError(f'Parameter {name} in hidet model does not find equivalent in PyTorch model.')


class PretrainedModelForCausalLM(nn.Module):
    def forward(self, input_ids: Tensor, position_ids: Tensor, attn_states: List[AttentionState]):
        """
        Forward run of the model.

        Parameters
        ----------
        input_ids: Tensor
            The input ids of the model.

        position_ids: Tensor
            The position ids of the model.

        attn_states: List[AttentionState]
            The attention states of the model.

        Returns
        -------
        hidden_states: Tensor
            The hidden states of the model.
        """
        raise NotImplementedError()

    def num_attention_layers(self):
        raise NotImplementedError()

    def num_attention_heads(self):
        raise NotImplementedError()

    def attention_head_size(self):
        return NotImplementedError()

    def embedding(self) -> Tensor:
        # with shape [hidden_size, vocab_size]
        raise NotImplementedError()

    def dtype(self) -> DataType:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls: Type, name: str, device='cuda', dtype=None, revision: Optional[str] = None):
        """
        Load a pretrained model from huggingface and convert it to hidet causal model.

        Parameters
        ----------
        name: str
            The name or path of the huggingface model.

        device: str
            The device of the model to be loaded to.

        dtype: str, optional
            The dtype of the model to be loaded to.

        revision: Optional[str]
            The revision of the model.

        Returns
        -------
        ret: PretrainedModelForCausalLM
            The loaded hidet model.
        """
        if not issubclass(cls, nn.Module):
            raise ValueError(f"{cls.__name__} should be a subclass of nn.Module")

        # load the pretrained huggingface model into cpu
        with torch.device("cuda"):  # reduce the time to load the model
            huggingface_token = hidet.option.get_option('tokens.for_huggingface')
            torch_model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float16, revision=revision, token=huggingface_token
            )

        torch_model = torch_model.cpu()
        torch.cuda.empty_cache()

        # create hidet model
        config = torch_model.config
        if dtype is None:
            if config.torch_dtype:
                assert isinstance(config.torch_dtype, torch.dtype)
                dtype = str(config.torch_dtype).rsplit('.', maxsplit=1)[-1]
            else:
                dtype = 'float16'
        hidet_model = cls(config)  # pylint: disable=too-many-function-args
        hidet_model.to(device=device, dtype=dtype)

        # copy the weights from torch model to hidet model
        copy_weights(torch_model, hidet_model)

        return hidet_model
