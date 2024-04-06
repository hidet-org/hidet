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
import ctypes
from typing import Dict, Tuple, List
from .utils import _marshal, _void_ptr_to


class SpecialTokenEntry(ctypes.Structure):
    _fields_ = [("token", ctypes.c_char_p), ("id", ctypes.c_uint32)]


class TemplateProcessingPostProcessorArgs(ctypes.Structure):
    _fields_ = [
        ("n_tmpl", ctypes.c_size_t),
        ("tmpl", ctypes.POINTER(ctypes.c_char_p)),
        ("n_special_tokens", ctypes.c_size_t),
        ("special_tokens", ctypes.POINTER(SpecialTokenEntry)),
    ]


def _postprocessor_args(config: Dict[str, any]) -> Tuple[ctypes.c_char_p, ctypes.c_void_p]:
    """
    Prepare arguments to make_model from the Hugging Face model configuration.
    The input should be parsed JSON.
    """
    type_ = config.get("type")

    if type_ == "ByteLevel":
        type_ = ctypes.c_char_p("ByteLevel".encode("utf-8"))
        return type_, ctypes.c_void_p()

    elif type_ == "TemplateProcessing":
        # Only support single processing for now
        tmpl: List[str] = []
        for item in config.get("single"):
            item = next(iter(item.values()))  # {a: {...}} -> {...}
            tmpl.append(item["id"])

        special_tokens: List[Tuple[str, int]] = []
        for token, info in config.get("special_tokens").items():
            if len(info["ids"]) != 1:
                raise ValueError("Multiple IDs for TemplateProcessing special tokens is not supported")
            special_tokens.append((token, info["ids"][0]))

        args = _marshal((len(tmpl), tmpl, len(special_tokens), special_tokens), TemplateProcessingPostProcessorArgs)
        return _marshal(type_), _void_ptr_to(args)
    else:
        raise ValueError(f"Unknown model type: {type_}")
