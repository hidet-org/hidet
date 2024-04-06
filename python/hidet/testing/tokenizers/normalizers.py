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
from typing import Dict, Tuple, Any

from .utils import _marshal, _void_ptr_to


class SequenceNormalizerArgs(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_size_t),
        ("types", ctypes.POINTER(ctypes.c_char_p)),
        ("child_args", ctypes.POINTER(ctypes.c_void_p)),
    ]


class PrependNormalizerArgs(ctypes.Structure):
    _fields_ = [("prefix", ctypes.c_char_p)]


class ReplaceNormalizerArgs(ctypes.Structure):
    _fields_ = [("from", ctypes.c_char_p), ("to", ctypes.c_char_p)]


def _normalizer_args(config: Dict[str, Any]) -> Tuple[ctypes.c_char_p, ctypes.c_void_p]:
    """
    Prepare arguments to make_normalizer from the Hugging Face normalizer configuration.
    The input should be parsed JSON.
    """
    typ = config.get("type")

    if typ == "Sequence":
        # Instantiate args for children
        children = config.get("normalizers", [])
        types, child_args = zip(*[_normalizer_args(child) for child in children])
        args = _marshal((len(children), types, child_args), SequenceNormalizerArgs)
        return _marshal(typ), _void_ptr_to(args)
    elif typ == "Prepend":
        prepend: str = config.get("prepend")
        args = _marshal((prepend,), PrependNormalizerArgs)
        return _marshal(typ), _void_ptr_to(args)
    elif typ == "Replace":
        pattern: dict = config.get("pattern")
        from_ = pattern["String"]
        to: str = config.get("content")
        args = _marshal((from_, to), ReplaceNormalizerArgs)
        return _marshal(typ), _void_ptr_to(args)
    else:
        raise ValueError(f"Unknown normalizer type: {typ}")
