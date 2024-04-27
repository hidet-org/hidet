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
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List, Tuple

from hidet.graph import Tensor


class ModelOutput(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.__class__ != ModelOutput and not is_dataclass(self):
            raise TypeError(f"{self.__module__}.{self.__class__} must be a dataclass to inherit from ModelOutput.")

    def __post_init__(self):
        """
        Called by dataclasses after initialization of dataclass values.

        Here to enable dict-like access
        """

        class_fields = fields(self)  # type: ignore
        if len(class_fields) == 0:
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any, ...]:
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: Tensor
    hidden_states: List[Tensor]


@dataclass
class BaseModelOutputWithPooling(BaseModelOutput):
    pooler_output: Tensor


@dataclass
class ImageClassifierOutput(BaseModelOutputWithPooling):
    logits: Tensor
