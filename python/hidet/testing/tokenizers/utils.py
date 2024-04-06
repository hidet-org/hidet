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


def _marshal(value, typ=None):
    """
    Marshal a Python value into a ctypes value (potentially recursively), using the given dtype as a hint.
    """
    if typ is not None and isinstance(value, typ):
        # Already marshalled
        return value
    elif isinstance(value, str) and (typ is None or typ == ctypes.c_char_p):
        # str -> null-terminated string
        return ctypes.c_char_p(value.encode("utf-8"))
    elif hasattr(value, "__iter__") and hasattr(typ, "_type_"):
        # iterable -> array pointer
        inner_typ = getattr(typ, "_type_")
        value = [_marshal(item, inner_typ) for item in value]
        return (inner_typ * len(value))(*value)
    elif isinstance(value, (list, tuple)) and hasattr(typ, "_fields_"):
        # iterable -> struct
        fields = []
        for item, (_, field_typ) in zip(value, getattr(typ, "_fields_")):
            fields.append(_marshal(item, field_typ))
        return typ(*fields)
    else:
        try:
            return typ(value)
        except TypeError as e:
            raise ValueError(f"Cannot marshal {value} ({type(value)}) into dtype {typ}") from e


def _void_ptr_to(value):
    """
    Utility function to create a void pointer to a ctypes value.
    """
    return ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
