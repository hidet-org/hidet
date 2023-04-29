from typing import Union, Optional, Sequence
from hidet.ir.type import DataType, tensor_type
from hidet.ir.expr import Expr
from hidet.ir.stmt import DeclareScope
from hidet.ir.layout import DataLayout
from hidet.lang.type_utils import shared_scope, register_scope

from hidet.ir.primitives.cpu import avx_f32x4_broadcast, avx_f32x4_fmadd, avx_f32x4_load, avx_f32x4_store, avx_f32x4_setzero