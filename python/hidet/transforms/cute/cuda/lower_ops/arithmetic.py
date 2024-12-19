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
from hidet.ir.expr import Expr, Var, call
from hidet.ir.type import DataType, TensorType, TensorPointerType, FuncType

from hidet.ir.cute.ops.arithmetic import Arithmetic, Fill
from hidet.ir.cute import TiledTensorLayout, TensorLayout, coalesce, flatten, size
from hidet.ir.cute.layout import group, filter
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

    def simplify(self, out_layout: TensorLayout, inp_layouts: List[TensorLayout]):
        dim_removed = []
        dim = len(out_layout.shape_tuple)
        for i in range(dim):
            can_be_removed = True
            for layout in inp_layouts:
                assert dim == len(layout.shape_tuple)
                if layout[i].stride != 0:
                    can_be_removed = False
                    break
            if can_be_removed:
                dim_removed.append(i)
        if len(dim_removed) == 0:
            return out_layout, inp_layouts

        def remove_dim(layout: TensorLayout):
            result_shape = [s for i, s in enumerate(layout.shape_tuple) if i not in dim_removed]
            result_stride = [d for i, d in enumerate(layout.stride_tuple) if i not in dim_removed]
            return TensorLayout(tuple(result_shape), tuple(result_stride))

        out_layout = remove_dim(out_layout)
        inp_layouts = [remove_dim(layout) for layout in inp_layouts]
        return out_layout, inp_layouts

    def emit(self, op: Arithmetic, args: List[Buffer], output: Buffer):
        assert all(isinstance(arg, Buffer) for arg in args)
        dst: Var = output.buffer
        from hidet.ir.cute.ops.arithmetic import local_broadcast, distributed_broadcast, broadcast_layout

        if isinstance(args[0].layout, TiledTensorLayout):
            layouts: List[TiledTensorLayout] = [arg.layout for arg in args]
            shapes: List[Tuple] = [layout.shape() for layout in layouts]
            _, _, val_layouts = distributed_broadcast(shapes, layouts)
            dst_val_layout = broadcast_layout(val_layouts)
        else:
            assert all(isinstance(arg.layout, TensorLayout) for arg in args)
            val_layouts: List[TensorLayout] = [arg.layout for arg in args]
            dst_val_layout = local_broadcast(val_layouts)

        def canonicalize(layout: TensorLayout):
            shape = list(flatten(layout.shape_tuple))
            stride = list(flatten(layout.stride_tuple))
            current = 1
            for i, (s, d) in enumerate(zip(shape, stride)):
                if d != 0:
                    stride[i] = current
                    current = current * s
            return coalesce(TensorLayout(tuple(shape), tuple(stride)))

        src_val_layouts = [canonicalize(layout) for layout in val_layouts]
        dst_val_layout = canonicalize(dst_val_layout)
        dst_val_layout, src_val_layouts = self.broadcast(dst_val_layout, src_val_layouts)
        dst_val_layout, src_val_layouts = self.simplify(dst_val_layout, src_val_layouts)

        func = op.op
        vector_size = 1
        if isinstance(func, Var):
            assert isinstance(func.type, FuncType)
            param_types = func.type.param_types
            output_ty = func.type.param_types[-1]
            if isinstance(output_ty, TensorType):
                vector_size = size(output_ty.shape)
            elif isinstance(output_ty, TensorPointerType):
                vector_size = size(output_ty.tensor_type.shape)

            def func_wrapper(x: List[Expr]):
                return call(func, x)

            apply = func_wrapper

        if vector_size == 1:
            extents = dst_val_layout.shape
            with self.for_grid(extents) as indices:
                srcs = [src.buffer[layout(indices, base=src.offset)] for src, layout in zip(args, src_val_layouts)]
                self.buffer_store(dst, [dst_val_layout(indices)], func(*srcs))
        else:
            src_val_layouts = [group(layout, vector_size) for layout in src_val_layouts]
            dst_val_layout = group(dst_val_layout, vector_size)

            _, outer = dst_val_layout
            extents = outer.shape
            with self.for_grid(extents) as indices:
                operands = tuple(
                    ~src.buffer[src_outer(indices, base=src.offset)]
                    for src, (_, src_outer), src_ty in zip(args, src_val_layouts, param_types[:-1])
                )
                operands += (~dst[outer(indices)],)
                self.append(apply(operands))


@register_impl(Fill)
class FillEmitter(OpEmitter):
    def emit(self, op: Fill, args: List[Buffer], output: Buffer):
        buf: Buffer = args[0]
        assert isinstance(buf.layout, TiledTensorLayout)
        dtype: DataType = buf.dtype
        val_layout = buf.layout.val_layout()
        val_layout = filter(val_layout)
        val_layout = TensorLayout(val_layout.shape_tuple)
        extents = flatten(val_layout.shape_tuple)
        with self.for_grid(extents) as indices:
            self.buffer_store(buf.buffer, [val_layout(indices, base=buf.offset)], dtype(op.val))
