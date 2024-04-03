import importlib
from dataclasses import astuple, dataclass
from typing import Dict

from transformers import PretrainedConfig


@dataclass
class RegistryEntry:
    """
    Configuration for dynamic loading of classes.

    We expect app directories to follow this file structure:

    apps/
    ├──<model_category>/
    │   ├── modeling/
    │   │   ├── <model_name>/
    │   │   │   ├── __init__.py
    │   │   └── ...
    │   ├── processing/
    │   │   ├── <processor_name>/
    │   │   │   ├── __init__.py
    │   │   └── ...
    ├──<model_category>/
        └── ...

    For example, model_category could be "image_classification", under which "resnet"
    is a model_name. The "resnet" module could contain class ResNetImageProcessor
    representing a callable for processing images.

    Use this to dynamically load pre-processors under a general naming scheme.
    """

    model_category: str
    module_name: str
    klass: str

    def __init__(self, model_category: str, module_name: str, klass: str):
        self.model_category = model_category
        self.module_name = module_name
        self.klass = klass


class Registry:
    module_registry: Dict[str, RegistryEntry] = {}

    @classmethod
    def load_module(cls, config: PretrainedConfig):
        architectures = getattr(config, "architectures")
        if not architectures:
            raise ValueError(f"Config {config.name_or_path} has no architecture.")

        # assume only 1 architecture available for now
        architecture = architectures[0]
        if architecture not in cls.module_registry:
            raise KeyError(
                f"No model class with architecture {architecture} found."
                f"Registered architectures: {', '.join(cls.module_registry.keys())}."
            )

        model_category, module_name, klass = astuple(cls.module_registry[architecture])

        module = importlib.import_module(f"hidet.apps.{model_category}.modeling.{module_name}")

        if klass not in dir(module):
            raise KeyError(f"No processor class named {klass} found in module {module}.")

        return getattr(module, klass)

    @classmethod
    def register(cls, arch: str, entry: RegistryEntry):
        cls.module_registry[arch] = entry
