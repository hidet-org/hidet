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
from typing import Dict, Tuple

from .utils import _marshal, _void_ptr_to


class SequenceDecoderArgs(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_size_t),
        ("types", ctypes.POINTER(ctypes.c_char_p)),
        ("child_args", ctypes.POINTER(ctypes.c_void_p)),
    ]


class ReplaceDecoderArgs(ctypes.Structure):
    _fields_ = [("pattern", ctypes.c_char_p), ("content", ctypes.c_char_p)]


class StripDecoderArgs(ctypes.Structure):
    _fields_ = [("content", ctypes.c_char_p), ("n_begin", ctypes.c_int), ("n_end", ctypes.c_int)]


def _decoder_args(config: Dict[str, any]) -> Tuple[ctypes.c_char_p, ctypes.c_void_p]:
    """
    Prepare arguments to make_model from the Hugging Face model configuration.
    The input should be parsed JSON.
    """
    type_ = config.get("type")

    if type_ == "Sequence":
        children = config.get("decoders", [])
        types, child_args = zip(*[_decoder_args(child) for child in children])
        args = _marshal((len(children), types, child_args), SequenceDecoderArgs)
        return _marshal(type_), _void_ptr_to(args)

    elif type_ == "Replace":
        pattern = config["pattern"]["String"]
        content = config["content"]
        args = _marshal((pattern, content), ReplaceDecoderArgs)
        return _marshal(type_), _void_ptr_to(args)

    elif type_ in ("ByteLevel", "Fuse", "ByteFallback"):
        return _marshal(type_), ctypes.c_void_p()

    elif type_ == "Strip":
        n_begin = config["start"]
        n_end = config["stop"]
        content = config["content"]
        args = _marshal((content, n_begin, n_end), StripDecoderArgs)
        return _marshal(type_), _void_ptr_to(args)

    else:
        raise ValueError(f"Unknown decoder type: {type_}")
