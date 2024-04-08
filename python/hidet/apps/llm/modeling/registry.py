from typing import Dict, Type, Tuple, Optional, List
import importlib

from .pretrained import PretrainedModelForCausalLM

model_registry: Dict[str, Tuple[str, str]] = {"LlamaForCausalLM": ("llama", "LlamaForCausalLM")}


def load_model_class(arch: str) -> Optional[Type[PretrainedModelForCausalLM]]:
    if arch not in model_registry:
        return None
    module_name, model_cls_name = model_registry[arch]
    module = importlib.import_module(f"hidet.apps.llm.modeling.{module_name}")
    return getattr(module, model_cls_name, None)


def supported_architectures() -> List[str]:
    return list(model_registry.keys())
