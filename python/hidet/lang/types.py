from hidet.ir.dtypes import i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bf16, tf32
from hidet.ir.dtypes import int8, int16, int32, int64, uint8, uint32, uint64, float16, float32, float64, bfloat16
from hidet.ir.dtypes import tfloat32

from hidet.ir.type import void_p, void, byte_p

from hidet.lang.constructs.declare import register_tensor, shared_tensor, tensor_pointer, tensor, DeclareScope