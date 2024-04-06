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

from .utils import _void_ptr_to, _marshal


class ByteLevelPreTokenizerArgs(ctypes.Structure):
    _fields_ = [("add_prefix_space", ctypes.c_bool), ("use_regex", ctypes.c_bool)]


def _pretokenizer_args(config: Dict[str, any]) -> Tuple[ctypes.c_char_p, ctypes.c_void_p]:
    type_ = config.get("type")

    if type_ == "ByteLevel":
        add_prefix_space = config.get("add_prefix_space", False)
        use_regex = config.get("use_regex", False)
        args = _marshal((add_prefix_space, use_regex), ByteLevelPreTokenizerArgs)
        return _marshal(type_), _void_ptr_to(args)
    else:
        raise ValueError(f"Unknown pre-tokenizer type: {type_}")
