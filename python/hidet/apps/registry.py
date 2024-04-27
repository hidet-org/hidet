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
import importlib
from dataclasses import astuple, dataclass
from typing import Dict


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
    def load_module(cls, architecture: str):
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
