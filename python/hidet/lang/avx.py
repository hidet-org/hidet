from typing import Union, Optional, Sequence
from hidet.ir.type import DataType, tensor_type
from hidet.ir.expr import Expr
from hidet.ir.stmt import DeclareScope
from hidet.ir.layout import DataLayout

from hidet.ir.primitives.cpu import avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load, avx_f32x4_store, avx_f32x4_setzero
from hidet.ir.primitives.cpu import avx_f32x8_broadcast, avx_f32x8_fmadd, avx_f32x8_load, avx_f32x8_store, avx_f32x8_setzero
from hidet.ir.primitives.cpu import avx_free, avx_malloc, x86_memcpy, x86_memset