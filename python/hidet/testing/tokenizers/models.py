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


class VocabEntry(ctypes.Structure):
    _fields_ = [("token", ctypes.c_char_p), ("id", ctypes.c_uint32)]


class MergeEntry(ctypes.Structure):
    _fields_ = [("first", ctypes.c_char_p), ("second", ctypes.c_char_p)]


class BPEModelArgs(ctypes.Structure):
    _fields_ = [
        ("n_vocab", ctypes.c_size_t),
        ("vocab", ctypes.POINTER(VocabEntry)),
        ("n_merges", ctypes.c_size_t),
        ("merges", ctypes.POINTER(MergeEntry)),
        ("byte_fallback", ctypes.c_bool),
    ]


def _model_args(config: Dict[str, any]) -> Tuple[ctypes.c_char_p, ctypes.c_void_p]:
    """
    Prepare arguments to make_model from the Hugging Face model configuration.
    The input should be parsed JSON.
    """
    type_ = config.get("type")

    if type_ == "BPE":
        vocab: Dict[str, int] = config.get("vocab")
        merges: List[Tuple[str, str]] = [m.split(" ") for m in config.get("merges")]
        byte_fallback = config.get("byte_fallback", False)
        args = _marshal((len(vocab), vocab.items(), len(merges), merges, byte_fallback), BPEModelArgs)
        return _marshal(type_), _void_ptr_to(args)
    else:
        raise ValueError(f"Unknown model type: {type_}")
