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
from typing import List, Union, Tuple
from hidet.ir.expr import Expr, Var

from hidet.ir.cute.ops.arithmetic import Arithmetic
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout, coalesce
from .registry import OpEmitter, Buffer, register_impl


@register_impl(Arithmetic)
class ArithmeticEmitter(OpEmitter):
    def broadcast_shape(self, x: Union[TensorLayout, List[int]], y: TensorLayout):
        if isinstance(x, list):
            x_shape = x
        else:
            x_shape = list(x.shape_tuple)
        y_shape = list(y.shape_tuple)
        shape = []
        i = 0
        j = 0
        while i < len(x_shape) or j < len(y_shape):
            if x_shape[i] < y_shape[j]:
                assert y_shape[j] % x_shape[i] == 0
                shape.append(x_shape[i])
                y_shape[j] //= x_shape[i]
                i += 1
            elif x_shape[i] > y_shape[j]:
                assert x_shape[i] % y_shape[j] == 0
                shape.append(y_shape[j])
                x_shape[i] //= y_shape[j]
                j += 1
            else:
                shape.append(x_shape[i])
                i += 1
                j += 1
        return shape

    def broadcast(self, out_layout: TensorLayout, inp_layouts: List[TensorLayout]):
        expanded_shape = self.broadcast_shape(out_layout, inp_layouts[0])
        for layout in inp_layouts:
            expanded_shape = self.broadcast_shape(expanded_shape, layout)

        def broadcast_layout(layout: TensorLayout, broadcast_shape: List[int]):
            shape = list(layout.shape_tuple)
            stride = list(layout.stride_tuple)
            result_shape = []
            result_stride = []
            i = 0
            j = 0
            while i < len(shape):
                if shape[i] > broadcast_shape[j]:
                    result_shape.append(broadcast_shape[j])
                    result_stride.append(stride[i])
                    shape[i] = shape[i] // broadcast_shape[j]
                    stride[i] *= broadcast_shape[j]
                    j += 1
                else:
                    assert shape[i] == broadcast_shape[j]
                    result_shape.append(shape[i])
                    result_stride.append(stride[i])
                    i += 1
                    j += 1
            return TensorLayout(tuple(result_shape), tuple(result_stride))

        out_layout = broadcast_layout(out_layout, expanded_shape)
        inp_layouts = [broadcast_layout(layout, expanded_shape) for layout in inp_layouts]
        return out_layout, inp_layouts

    def emit(self, op: Arithmetic, args: List[Buffer], output: Buffer):
        assert all(isinstance(arg, Buffer) for arg in args)
        srcs: List[Expr] = [arg.buffer for arg in args]
        dst: Var = output.buffer
        layouts: List[TiledTensorLayout] = [arg.layout for arg in args]
        shapes: List[Tuple] = [layout.shape() for layout in layouts]
        _, _, val_layouts = op.broadcast(shapes, layouts)
        dst_val_layout = op.deduce_broadcast_layout(val_layouts)

        def canonicalize(layout: TensorLayout):
            shape = list(layout.shape_tuple)
            stride = list(layout.stride_tuple)
            current = 1
            for i, (s, d) in enumerate(zip(shape, stride)):
                if d != 0:
                    stride[i] = current
                    current = current * s
            return coalesce(TensorLayout(tuple(shape), tuple(stride)))

        src_val_layouts = [canonicalize(layout) for layout in val_layouts]
        dst_val_layout = canonicalize(dst_val_layout)
        dst_val_layout, src_val_layouts = self.broadcast(dst_val_layout, src_val_layouts)

        extents = dst_val_layout.shape
        with self.for_grid(extents) as indices:
            srcs = [src[layout(indices)] for src, layout in zip(srcs, src_val_layouts)]
            self.assign(dst[dst_val_layout(indices)], op.op(*srcs))
