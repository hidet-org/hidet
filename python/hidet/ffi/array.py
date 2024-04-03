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
from typing import Sequence
import ctypes
import struct
from hidet.ir.type import void_p
from hidet.ir.dtypes import i8, i16, i32, i64, u8, u16, u32, u64, f32, f64


class Array:
    format_character = {
        void_p: 'P',
        i8: 'b',
        i16: 'h',
        i32: 'i',
        i64: 'q',
        u8: 'B',
        u16: 'H',
        u32: 'I',
        u64: 'Q',
        f32: 'f',
        f64: 'd',
    }

    def __init__(self, base_type, length: int):
        self.item_format: str = self.format_character[base_type]
        self.item_nbytes: int = struct.calcsize(self.item_format)
        self.format: str = str(length) + self.item_format
        self.nbytes: int = struct.calcsize(self.format)
        self.buffer = bytearray(self.nbytes)
        self.length: int = length

    def __getitem__(self, item):
        return struct.unpack_from(self.item_format, self.buffer, item * self.item_nbytes)[0]

    def __setitem__(self, key, value):
        return struct.pack_into(self.item_format, self.buffer, key * self.item_nbytes, value)

    def __iter__(self):
        return iter(v[0] for v in struct.iter_unpack(self.item_format, self.buffer))

    def __len__(self):
        return self.length

    def data_ptr(self):
        char_array = (ctypes.c_char * self.nbytes).from_buffer(self.buffer)
        return ctypes.cast(char_array, ctypes.c_void_p)

    @staticmethod
    def from_int_list(data: Sequence):
        array = Array(i32, len(data))
        struct.pack_into(array.format, array.buffer, 0, *data)
        return array
