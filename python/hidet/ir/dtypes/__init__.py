from .float32 import float32, f32
from .tfloat32 import tfloat32, tf32
from .bfloat16 import bfloat16, bf16
from .float16 import float16, f16

from .int64 import int64, i64
from .int32 import int32, i32
from .int16 import int16, i16
from .int8 import int8, i8

from .uint64 import uint64, u64
from .uint32 import uint32, u32
from .uint16 import uint16, u16
from .uint8 import uint8, u8

from .boolean import boolean

name2dtype = {
    'float32': float32,
    'tfloat32': tfloat32,
    'bfloat16': bfloat16,
    'float16': float16,
    'int64': int64,
    'int32': int32,
    'int16': int16,
    'int8': int8,
    'uint64': uint64,
    'uint32': uint32,
    'uint16': uint16,
    'uint8': uint8,
    'bool': boolean,
}

sname2dtype = {
    'f32': float32,
    'tf32': tfloat32,
    'bf16': bfloat16,
    'f16': float16,
    'i64': int64,
    'i32': int32,
    'i16': int16,
    'i8': int8,
    'u64': uint64,
    'u32': uint32,
    'u16': uint16,
    'u8': uint8,
    'bool': boolean,
}
