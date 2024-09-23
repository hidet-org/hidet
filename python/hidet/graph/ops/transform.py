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
from typing import List, Optional, Tuple, Union, Sequence
from hidet.ir.type import DataType, data_type
from hidet.ir.expr import Expr, Constant, if_then_else, convert, cast as ir_cast, is_constant, logical_or
from hidet.ir.expr import Int
from hidet.ir.layout import RowMajorLayout
from hidet.ir.utils import index_deserialize, index_serialize
from hidet.utils import prod
from hidet.ir.cute import TensorLayout, compact_row_major, coalesce
from .utils import Task, InverseMap, Operator, Tensor, TensorNode, compute, input_like, normalize_dim, can_broadcast
from .utils import TensorInput, normalize_slice
from .transpose2d import TransposeOp2D


def is_true(x: Union[Expr, bool]) -> bool:
    if isinstance(x, (Constant, bool)):
        return bool(x) is True
    return False


def same_shape(shape_a: Sequence[Union[Expr, int]], shape_b: Sequence[Union[Expr, int]]) -> bool:
    return len(shape_a) == len(shape_b) and all(a == b for a, b in zip(shape_a, shape_b))


class ReshapeTask(Task):
    def __init__(self, x: TensorNode, y_shape: List[Int]):
        y_shape = self.normalize_shape(x.shape, y_shape)
        if not isinstance(x.type.layout, RowMajorLayout):
            raise NotImplementedError(
                'currently, only support row major layout. Please use '
                '.contiguous() to transfer the given tensor into row major layout first.'
            )

        def index_map(dst_indices, src_shape, dst_shape):
            src_groups = []
            dst_groups = []
            if any(isinstance(v, Expr) for v in list(src_shape) + list(dst_shape)):
                src_groups = [list(range(len(src_shape)))]
                dst_groups = [list(range(len(dst_shape)))]
            else:
                i, j = 0, 0
                while i < len(src_shape) and j < len(dst_shape):
                    src_group = [i]
                    dst_group = [j]
                    x_size, y_size = src_shape[i], dst_shape[j]
                    i += 1
                    j += 1
                    while x_size != y_size:
                        if x_size < y_size:
                            x_size *= src_shape[i]
                            src_group.append(i)
                            i += 1
                        else:
                            y_size *= dst_shape[j]
                            dst_group.append(j)
                            j += 1
                    src_groups.append(src_group)
                    dst_groups.append(dst_group)
                if i < len(src_shape):
                    src_groups.append(list(range(i, len(src_shape))))
                    dst_groups.append([])
                if j < len(dst_shape):
                    src_groups.append([])
                    dst_groups.append(list(range(j, len(dst_shape))))
            src_indices = []
            for src_group, dst_group in zip(src_groups, dst_groups):
                x_group_shape = [src_shape[r] for r in src_group]
                y_group_shape = [dst_shape[r] for r in dst_group]
                y_group_indices = [dst_indices[r] for r in dst_group]
                x_group_indices = index_deserialize(index_serialize(y_group_indices, y_group_shape), x_group_shape)
                src_indices.extend(x_group_indices)
            assert len(src_indices) == len(src_shape)
            return src_indices

        def inverse_map(*x_indices):
            return index_map(x_indices, src_shape=y_shape, dst_shape=x.shape)

        y = compute(
            name='y',
            shape=y_shape,
            fcompute=lambda *indices: x[index_map(indices, src_shape=x.shape, dst_shape=y_shape)],
        )
        inverse_map = InverseMap.from_lambda(inverse_map, num_args=len(x.shape))
        inverse_map.tile_mapping = TensorLayout(1)
        super().__init__(
            name='reshape', inputs=[x], outputs=[y], inverse_map={x: inverse_map}, attributes={'shape': y_shape}
        )

    def normalize_shape(self, origin_shape: Sequence[Int], shape: Sequence[Int]):
        # [1, 3, 224, 224], [1, -1, 224, 0] => [1, 3, 224, 224]
        shape = list(shape)
        for i in range(len(shape)):
            if isinstance(shape[i], int) and shape[i] == 0:
                if i >= len(origin_shape):
                    raise ValueError(
                        '0 is used outside original shape: ' 'origin {} target {}'.format(origin_shape, shape)
                    )
                shape[i] = origin_shape[i]

        size = prod(origin_shape)
        cnt = sum(1 for v in shape if isinstance(v, int) and v == -1)
        if cnt == 0:
            total = prod(shape)
            self._assert(
                total == size,
                (
                    'Reshape: given shape has different size with input tensor: '
                    'shape {} and size {}'.format(shape, size)
                ),
            )
            return shape
        elif cnt == 1:
            remain_size = prod([v for v in shape if not is_constant(v) or v != -1])
            self._assert(
                size % remain_size == 0,
                'Given shape is incompatible with input tensor: ' 'shape {} and size {}'.format(shape, size),
            )
            return [v if not is_constant(v) or v != -1 else size // remain_size for v in shape]
        else:
            raise ValueError('Can not infer the shape when there are multiple -1: {}'.format(shape))


class RearrangeTask(Task):
    def __init__(self, x: TensorNode, plan: List[List[int]]):
        x_shape = x.shape
        y_shape = [prod([x_shape[i] for i in dims]) for dims in plan]

        def index_split(total_index, dim_sizes: List[Expr]) -> List:
            bases = [prod(dim_sizes[i + 1 :]) for i in range(len(dim_sizes))]
            return [(total_index // base) % dim for dim, base in zip(dim_sizes, bases)]

        def fcompute(*y_indices):
            x_indices = [None for _ in range(len(x_shape))]
            for i, y_index in enumerate(y_indices):
                dims = plan[i]
                if len(dims) == 0:
                    # this new dimension has size 1
                    continue
                split_indices = index_split(total_index=y_index, dim_sizes=[x_shape[k] for k in dims])
                for j, x_index in zip(dims, split_indices):
                    x_indices[j] = x_index
            for i, x_index in enumerate(x_indices):
                if x_index is None:
                    if isinstance(x_shape[i], Constant) and int(x_shape[i]) > 1:
                        msg = 'Rearrange plan {} on tensor {} leave non-one dimension {} not been accessed'.format(
                            plan, x_shape, i
                        )
                        raise ValueError(msg)
                    else:
                        x_indices[i] = 0
            return x[x_indices]

        y = compute(name='y', shape=y_shape, fcompute=fcompute)

        def inverse_map(*x_indices):
            y_indices = []
            for dims in plan:
                cnt = convert(0)
                for dim in dims:
                    cnt = cnt * x_shape[dim] + x_indices[dim]
                y_indices.append(cnt)
            return y_indices

        inverse_map = InverseMap.from_lambda(inverse_map, len(x_shape))
        if any(not is_constant(v) for v in x_shape):  # dynamic shape
            dim_set = set()
            for dims in plan:
                for i in dims:
                    dim_set.add(i)
            if all(i in dim_set or (is_constant(v) and v == 1) for i, v in enumerate(x_shape)):
                inverse_map.tile_mapping = TensorLayout(1)
            else:
                inverse_map.tile_mapping = None
        elif len(plan) == 0 or all(len(dims) == 0 for dims in plan):  # empty tensor
            inverse_map.tile_mapping = None
        else:
            x_shape_value = [int(v) for v in x_shape]
            from hidet.ir import cute

            dst_shape = cute.flatten(tuple(tuple(x_shape_value[i] for i in dims) for dims in plan))
            dst_stride_prime = compact_row_major(dst_shape)
            dst_stride = [0] * len(x_shape_value)
            idx = 0
            for dims in plan:
                for i in dims:
                    dst_stride[i] = dst_stride_prime[idx]
                    idx += 1

            inverse_map.tile_mapping = coalesce(
                TensorLayout(tuple(reversed(x_shape_value)), tuple(reversed(dst_stride)))
            )
        super().__init__(name='rearrange', inputs=[x], outputs=[y], inverse_map={x: inverse_map})
        self.plan = plan


class ConcatTask(Task):
    def __init__(self, inputs: List[TensorNode], axis: int):
        shapes: List[List[Expr]] = [t.shape for t in inputs]
        n = len(shapes)
        assert n > 0
        for i in range(1, n):
            if len(shapes[0]) != len(shapes[i]):
                raise ValueError('Concat: all shapes must have the same rank, got {}'.format(shapes))
            if any(is_constant(a, b) and a != b for j, (a, b) in enumerate(zip(shapes[0], shapes[i])) if j != axis):
                raise ValueError(
                    'Concat: all tensors must have the same shape except axis dimension, '
                    'got {}, axis {}'.format(shapes, axis)
                )
        rank = len(shapes[0])
        out_shape = [shapes[0][i] if i != axis else sum(shapes[j][i] for j in range(n)) for i in range(rank)]

        def fmap(*indices):
            pre_sum = [sum(shapes[j][axis] for j in range(i)) for i in range(n + 1)]
            value = inputs[-1][indices[:axis] + (indices[axis] - pre_sum[-2],) + indices[axis + 1 :]]
            for i, inp in reversed(list(zip(range(n - 1), inputs[: n - 1]))):
                input_i_value = inp[indices[:axis] + (indices[axis] - pre_sum[i],) + indices[axis + 1 :]]
                value = if_then_else(indices[axis] < pre_sum[i + 1], input_i_value, value)
            return value

        out = compute(name='out', shape=out_shape, fcompute=lambda *indices: fmap(*indices))

        super().__init__(name='concat', inputs=inputs, outputs=[out])


class TakeTask(Task):
    def __init__(self, data: TensorNode, indices: TensorNode, axis=0):
        output_shape = data.shape[:axis] + indices.shape + data.shape[axis + 1 :]
        assert 0 <= axis < len(data.shape)

        def fmap(*output_indices):
            indices_indices = output_indices[axis : axis + len(indices.shape)]
            index_value = indices[indices_indices]
            index_value = if_then_else(index_value < 0, index_value + data.shape[axis], index_value)
            data_indices = output_indices[:axis] + (index_value,) + output_indices[axis + len(indices.shape) :]
            return data[data_indices]

        output = compute(name='output', shape=output_shape, fcompute=lambda *output_indices: fmap(*output_indices))
        super().__init__(name='take', inputs=[data, indices], outputs=[output])


class GatherTask(Task):
    def __init__(self, data: TensorInput, indices: TensorInput, axis=0):
        output_shape = data.shape[:axis] + [indices.shape[axis]] + data.shape[axis + 1 :]

        def fmap(*output_indices):
            index_value = indices[output_indices]
            index_value = if_then_else(index_value < 0, index_value + data.shape[axis], index_value)
            data_indices = output_indices[:axis] + (index_value,) + output_indices[axis + 1 :]
            return data[data_indices]

        output = compute(name='output', shape=output_shape, fcompute=lambda *output_indices: fmap(*output_indices))
        super().__init__(name='gather', inputs=[data, indices], outputs=[output])


class IdxSelTask(Task):
    def __init__(self, data: TensorInput, index: TensorInput, dim=0):
        output_shape = data.shape[:dim] + [index.shape[0]] + data.shape[dim + 1 :]

        def fmap(*output_indices):
            index_value = index[output_indices[dim]]
            index_value = if_then_else(index_value < 0, index_value + data.shape[dim], index_value)
            data_indices = output_indices[:dim] + (index_value,) + output_indices[dim + 1 :]
            return data[data_indices]

        output = compute(name='output', shape=output_shape, fcompute=lambda *output_indices: fmap(*output_indices))
        super().__init__(name='idxsel', inputs=[data, index], outputs=[output])


class RepeatInterleaveTask(Task):
    def __init__(self, data: TensorInput, repeats: int, dim: int = None):
        # NOTE: Currently we only support the case where `repeats` is int, as that is what we actually encountered
        # while benchmarking models.
        if dim is None:
            # if dim is None: flatten the input tensor and repeat the whole tensor
            output_shape = [prod(data.shape) * repeats]
        else:
            output_shape = data.shape[:dim] + [data.shape[dim] * repeats] + data.shape[dim + 1 :]

        def fmap(*output_indices):
            if dim is None:
                idx = output_indices[0] // repeats
                return data[index_deserialize(idx, data.shape)]
            else:
                idx = output_indices[dim] // repeats
                return data[output_indices[:dim] + (idx,) + output_indices[dim + 1 :]]

        out = compute(name='out', shape=output_shape, fcompute=lambda *indices: fmap(*indices))

        super().__init__(name='repeat_interleave', inputs=[data], outputs=[out])


class StridedSliceTask(Task):
    def __init__(
        self,
        data: TensorNode,
        starts: List[Optional[int]],
        ends: List[Optional[int]],
        axes: List[int],
        strides: List[int],
    ):
        assert len(starts) == len(ends) == len(axes) == len(strides)
        if len(axes) != len(set(axes)):
            raise ValueError('Duplicated axes in slice, axes: {}'.format(axes))
        output_shape = list(data.shape)
        axis2info = {}
        for axis, start, end, stride in zip(axes, starts, ends, strides):
            if stride == 0:
                raise NotImplementedError(
                    'Stride can not be 0 in slicing: '
                    'starts {} ends {} axes {} strides {}.'.format(starts, ends, axes, strides)
                )
            if stride > 0:
                output_shape[axis] = (end - start + stride - 1) // stride
            else:
                output_shape[axis] = (start - end + (-stride) - 1) // (-stride)
            if is_constant(output_shape[axis]) and output_shape[axis] < 0:
                raise NotImplementedError(
                    'Slice result can not be: '
                    'starts {} ends {} axes {} strides {}'.format(starts, ends, axes, strides)
                )
            axis2info[axis] = (start, end, stride)

        def fmap(indices):
            data_indices = []
            for axis, index in enumerate(indices):
                if axis in axis2info:
                    start, _, stride = axis2info[axis]
                    data_indices.append(start + index * stride)
                else:
                    data_indices.append(index)
            return data[data_indices]

        out = compute('out', shape=output_shape, fcompute=lambda *indices: fmap(indices))
        super().__init__(name='slice', inputs=[data], outputs=[out])


class BroadcastTask(Task):
    def __init__(self, data: TensorNode, shape: List[int]):
        data_shape = data.shape
        if not can_broadcast(data_shape, shape):
            raise ValueError('Can not broadcast a tensor with shape {} to {}'.format(data_shape, shape))

        def fmap(*indices):
            expanded = len(shape) - len(data_shape)
            indices = indices[expanded:]
            indices = [if_then_else(data_shape[i] != 1, v, 0) for i, v in enumerate(indices)]
            return data[indices]

        out = compute('out', shape=shape, fcompute=fmap)
        super().__init__(name='broadcast', inputs=[data], outputs=[out])


class PadTask(Task):
    def __init__(self, data: TensorNode, pads: List[int], value: float):
        from hidet.ir.compute import cops

        out = cops.pad(data, pads, value)
        super().__init__(name='pad', inputs=[data], outputs=[out])


class TileTask(Task):
    def __init__(self, data: TensorNode, repeats: Sequence[int]):
        assert len(data.shape) == len(repeats)
        out_shape = [a * b for a, b in zip(data.shape, repeats)]

        def fmap(*indices):
            indices = [idx % data.shape[i] for i, idx in enumerate(indices)]
            return data[indices]

        out = compute(name='out', shape=out_shape, fcompute=fmap)
        super().__init__(name='tile', inputs=[data], outputs=[out])


class TriuTask(Task):
    def __init__(self, x: TensorNode, diagonal: Int):
        dtype = x.type.dtype

        def fmap(*indices):
            *ks, i, j = indices
            return if_then_else(j >= i + diagonal, x[(*ks, i, j)], dtype.zero)

        out = compute(name='out', shape=x.shape, fcompute=fmap)
        super().__init__(name='triu', inputs=[x], attributes={'diagonal': diagonal}, outputs=[out])


class TrilTask(Task):
    def __init__(self, x: TensorNode, diagonal: Int):
        dtype = x.type.dtype

        def fmap(*indices):
            *ks, i, j = indices
            return if_then_else(j <= i + diagonal, x[(*ks, i, j)], dtype.zero)

        out = compute(name='out', shape=x.shape, fcompute=fmap)
        super().__init__(name='tril', inputs=[x], attributes={'diagonal': diagonal}, outputs=[out])


class Im2ColTask(Task):
    def __init__(
        self, x: TensorNode, kernel_size: List[Int], dilation: List[Int], padding: List[Int], stride: List[Int]
    ):
        dtype = x.type.dtype
        batch_size = x.shape[0]
        n_input_plane = x.shape[1]
        input_height = x.shape[2]
        input_width = x.shape[3]

        output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) // stride[0] + 1
        output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) // stride[1] + 1
        n_output_plane = n_input_plane * kernel_size[0] * kernel_size[1]
        output_length = output_height * output_width

        output_shape = [batch_size, n_output_plane, output_length]

        def fmap(*indices):
            b_i, ck_i, o_i = indices
            c_i = ck_i // prod(kernel_size)
            ck_i %= prod(kernel_size)
            k_hi = ck_i // kernel_size[1]
            k_wi = ck_i % kernel_size[1]
            o_hi = o_i // output_width
            o_wi = o_i % output_width

            i_hi = o_hi * stride[0] - padding[0] + k_hi * dilation[0]
            i_wi = o_wi * stride[1] - padding[1] + k_wi * dilation[1]
            return if_then_else(
                logical_or(i_hi < 0, i_wi < 0, i_hi >= input_height, i_wi >= input_width),
                dtype.zero,
                x[b_i, c_i, i_hi, i_wi],
            )

        out = compute(name='out', shape=output_shape, fcompute=fmap)

        super().__init__(
            name='im2col',
            inputs=[x],
            attributes={'kernel_size': kernel_size, 'dilation': dilation, 'padding': padding, 'stride': stride},
            outputs=[out],
        )


class AsStridedTask(Task):
    def __init__(
        self,
        x: TensorNode,
        size: Union[int, List[int]],
        stride: Union[int, List[int]],
        storage_offset: Optional[int] = None,
    ):
        def unravel_index(index, shape):
            out = []
            for s in reversed(shape):
                out.append(index % s)
                index = index // s
            return list(reversed(out))

        storage_shift = storage_offset if storage_offset is not None else 0

        def fmap(*indices):
            stride1d = 0
            for i, s in zip(indices, stride):
                stride1d += i * s
            stride1d += storage_shift

            new_indices = unravel_index(stride1d, x.shape)
            return x[new_indices]

        out = compute(name='out', shape=size, fcompute=fmap)

        super().__init__(
            name='as_strided',
            inputs=[x],
            attributes={'size': size, 'stride': stride, 'storage_offset': storage_offset},
            outputs=[out],
        )


class FlipTask(Task):
    def __init__(self, x: Tensor, dims: Union[List[int], Tuple[int]]):
        def fmap(*indices):
            idx = []
            for i in range(len(indices)):
                idx.append(if_then_else(i in dims, x.shape[i] - indices[i] - 1, indices[i]))
            return x[idx]

        out = compute(name='out', shape=x.shape, fcompute=fmap)
        super().__init__(name='flip', inputs=[x], attributes={'dims': dims}, outputs=[out])


class ReshapeOp(Operator):
    def __init__(self, x: Tensor, shape):
        task = ReshapeTask(input_like(x, 'x'), shape)
        super().__init__(inputs=[x], attributes={'shape': shape}, task=task)


class RearrangeOp(Operator):
    def __init__(self, x: Tensor, plan: List[List[int]]):
        super().__init__(inputs=[x], attributes={'plan': plan}, task=RearrangeTask(input_like(x, 'x'), plan=plan))


class SqueezeOp(Operator):
    def __init__(self, x: Tensor, dims: List[int]):
        super().__init__(
            inputs=[x],
            attributes={'dims': dims},
            task=RearrangeTask(input_like(x, 'x'), plan=[[i] for i in range(len(x.shape)) if i not in dims]),
        )


class UnsqueezeOp(Operator):
    def __init__(self, x: Tensor, dims: List[int]):
        dims = list(dims)
        plan = []
        c = 0
        for i in range(len(x.shape) + len(dims)):
            if i in dims:
                plan.append([])
            else:
                plan.append([c])
                c += 1
        if c != len(x.shape):
            raise ValueError('Invalid unsqueeze dims: {} for shape: {}'.format(dims, x.shape))
        super().__init__(inputs=[x], attributes={'dims': dims}, task=RearrangeTask(input_like(x, 'x'), plan=plan))


class FlattenOp(Operator):
    def __init__(self, x: Tensor, start_dim: int, end_dim: int):
        rank = len(x.shape)
        start_dim = normalize_dim(start_dim, rank)
        end_dim = normalize_dim(end_dim, rank)
        assert 0 <= start_dim <= end_dim < rank
        dims = list(range(len(x.shape)))
        plan = [[v] for v in dims[:start_dim]] + [dims[start_dim : end_dim + 1]] + [[v] for v in dims[end_dim + 1 :]]
        super().__init__(
            inputs=[x],
            attributes={'start_dim': start_dim, 'end_dim': end_dim},
            task=RearrangeTask(input_like(x, 'x'), plan=plan),
        )


class PermuteDimsOp(Operator):
    def __init__(self, x: Tensor, axes: Optional[List[int]] = None):
        if axes and len(axes) != len(x.shape):
            msg = 'Transpose tensor with shape {} expect a permutation of axes with length {}, got {}'.format(
                x.shape, len(x.shape), axes
            )
            raise ValueError(msg)
        if axes is None:
            axes = list(reversed(range(len(x.shape))))
        plan = [[v] for v in axes]
        super().__init__(inputs=[x], attributes={'axes': axes}, task=RearrangeTask(input_like(x, 'x'), plan))


class CastOp(Operator):
    def __init__(self, x: Tensor, dtype: DataType):
        from .arithmetic import UnaryElementwiseTask

        super().__init__(
            inputs=[x],
            attributes={'dtype': dtype},
            task=UnaryElementwiseTask('cast', input_like(x, 'x'), op=lambda v: ir_cast(v, dtype)),
        )


class ConcatOp(Operator):
    def __init__(self, *tensors: Tensor, axis: int):
        tensors = list(tensors)
        if len(tensors) == 0:
            raise ValueError('Concat requires at least one tensor, 0 given.')
        tensors = [tensor for tensor in tensors if tensor.shape != (0,)] or [tensors[0]]
        axis = normalize_dim(axis, len(tensors[0].shape))
        super().__init__(
            inputs=tensors,
            attributes={'axis': axis},
            task=ConcatTask([input_like(tensor, 'x{}'.format(idx)) for idx, tensor in enumerate(tensors)], axis=axis),
        )


class TakeOp(Operator):
    def __init__(self, data: Tensor, indices: Tensor, axis: int):
        super().__init__(
            inputs=[data, indices],
            attributes={'axis': axis},
            task=TakeTask(input_like(data, 'data'), input_like(indices, 'indices'), axis=axis),
        )


class GatherOp(Operator):
    def __init__(self, data: Tensor, indices: Tensor, axis: int):
        axis = normalize_dim(axis, rank=len(data.shape))
        super().__init__(
            inputs=[data, indices],
            attributes={'axis': axis},
            task=GatherTask(input_like(data, 'data'), input_like(indices, 'indices'), axis=axis),
        )


class IdxSelOp(Operator):
    def __init__(self, data: Tensor, index: Tensor, dim: int):
        super().__init__(
            inputs=[data, index],
            attributes={'dim': dim},
            task=IdxSelTask(input_like(data, 'data'), input_like(index, 'index'), dim=dim),
        )


class RepeatInterleaveOp(Operator):
    def __init__(self, data: Tensor, repeats: int, dim: int = None):
        super().__init__(
            inputs=[data],
            attributes={'repeats': repeats, 'dim': dim},
            task=RepeatInterleaveTask(input_like(data, 'data'), repeats, dim=dim),
        )


class StridedSliceOp(Operator):
    def __init__(
        self,
        data: Tensor,
        starts: Sequence[Optional[int]],
        ends: Sequence[Optional[int]],
        axes: Optional[Sequence[Optional[int]]] = None,
        strides: Optional[Sequence[Optional[int]]] = None,
    ):
        starts, ends, axes, strides = normalize_slice(data.shape, starts, ends, axes, strides)
        task = StridedSliceTask(input_like(data, 'data'), starts, ends, axes, strides)
        super().__init__(
            inputs=[data], attributes={'starts': starts, 'ends': ends, 'axes': axes, 'strides': strides}, task=task
        )


class BroadcastOp(Operator):
    def __init__(self, data: Tensor, shape: List[int]):
        super().__init__(
            inputs=[data], attributes={'shape': shape}, task=BroadcastTask(input_like(data, 'data'), shape)
        )


class PadOp(Operator):
    def __init__(self, data: Tensor, pads: List[int], mode: str = 'constant', value: float = 0.0):
        if len(pads) < len(data.shape) * 2:
            assert len(pads) % 2 == 0, 'The pads must have even number of elements.'
            half = len(pads) // 2
            extra = [0 for _ in range(len(data.shape) - half)]
            pads = extra + pads[:half] + extra + pads[half:]
        if mode != 'constant':
            raise NotImplementedError("Padding mode '{}' has not been implemented yet.".format(mode))
        super().__init__(
            inputs=[data],
            attributes={'pads': pads, 'mode': mode, 'value': value},
            task=PadTask(input_like(data, 'data'), pads, value),
        )


class TileOp(Operator):
    def __init__(self, data: Tensor, repeats: Sequence[int]):
        if len(repeats) != len(data.shape):
            raise ValueError(
                "The length of 'repeats' parameter of Tile operator expects to have the "
                "same length as data shape. shape: {}, repeats: {}".format(data.shape, repeats)
            )
        super().__init__(
            inputs=[data], attributes={'repeats': repeats}, task=TileTask(input_like(data, 'data'), repeats)
        )


class TriuOp(Operator):
    def __init__(self, x: Tensor, diagonal: Int = 0):
        super().__init__(inputs=[x], attributes={'diagonal': diagonal}, task=TriuTask(input_like(x, 'x'), diagonal))


class TrilOp(Operator):
    def __init__(self, x: Tensor, diagonal: Int = 0):
        super().__init__(inputs=[x], attributes={'diagonal': diagonal}, task=TrilTask(input_like(x, 'x'), diagonal))


class Im2ColOp(Operator):
    def __init__(self, x: Tensor, kernel_size: List[Int], dilation: List[Int], padding: List[Int], stride: List[Int]):
        super().__init__(
            inputs=[x],
            attributes={'kernel_size': kernel_size, 'dilation': dilation, 'padding': padding, 'stride': stride},
            task=Im2ColTask(input_like(x, 'x'), kernel_size, dilation, padding, stride),
        )


class AsStridedOp(Operator):
    def __init__(
        self,
        x: Tensor,
        size: Union[int, List[int]],
        stride: Union[int, List[int]],
        storage_offset: Optional[int] = None,
    ):
        super().__init__(
            inputs=[x],
            attributes={'size': size, 'stride': stride, 'storage_offset': storage_offset},
            task=AsStridedTask(input_like(x, 'x'), size, stride, storage_offset),
        )


class FlipOp(Operator):
    def __init__(self, x: Tensor, dims: Union[List[int], Tuple[int]]):
        super().__init__(inputs=[x], attributes={'dims': dims}, task=FlipTask(input_like(x, 'x'), dims))


def reshape(x: Tensor, shape) -> Tensor:
    if same_shape(x.shape, shape):
        return x
    return ReshapeOp(x, shape).outputs[0]


def rearrange(x: Tensor, plan: List[List[int]]) -> Tensor:
    """Rearrange a tensor. This task is a general task of squeeze, unsqueeze, flatten, and perm.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    plan: List[List[int]]
        The rearrange plan.

    Returns
    -------
    ret: Tensor
        The task to conduct rearrangement.

    Examples
    --------
    - squeeze([1, 1, 2, 3], dims=[0, 1]) = rearrange([1, 1, 2, 3], plan=[[2], [3]]) => Tensor([2, 3])
    - unsqueeze([2, 3], dims=[0, 1]) = rearrange([2, 3], plan=[[], [], [0], [1]]) => Tensor([1, 1, 2, 3])
    - flatten([2, 3, 4, 5], start_dim=1, end_dim=2) = rearrange([2, 3, 4, 5], plan=[[0], [1, 2], [3]]) =>
      Tensor([2, 12, 5])
    """
    if not isinstance(plan, (list, tuple)) or any(not isinstance(v, (list, tuple)) for v in plan):
        raise ValueError('plan should be List[List[int]], but got: {}'.format(plan))
    return RearrangeOp(x, plan).outputs[0]


def squeeze(x: Tensor, dims: Union[int, Sequence[int]]) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    if len(dims) == 0:
        return x
    return SqueezeOp(x, dims).outputs[0]


def unsqueeze(x: Tensor, dims: Union[int, Sequence[int]]) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    dims = [normalize_dim(dim, len(x.shape) + len(dims)) for dim in dims]
    if len(dims) == 0:
        return x
    return UnsqueezeOp(x, dims).outputs[0]


def flatten(x: Tensor, start_dim=0, end_dim=-1) -> Tensor:
    start_dim = normalize_dim(start_dim, len(x.shape))
    end_dim = normalize_dim(end_dim, len(x.shape))
    if start_dim >= end_dim:
        return x
    return FlattenOp(x, start_dim, end_dim).outputs[0]


def transpose(x: Tensor, axes: Optional[Sequence[int]] = None) -> Tensor:
    rank = len(x.shape)
    if rank == 2:
        return TransposeOp2D(x).outputs[0]
    if axes is None:
        axes = list(reversed(range(rank)))
    axes = [normalize_dim(dim, rank) for dim in axes]
    dims = []
    i = 0
    for j in range(rank):
        if j in axes:
            dims.append(axes[i])
            i += 1
        else:
            dims.append(j)
    return PermuteDimsOp(x, dims).outputs[0]


def permute_dims(x: Tensor, /, axes: Sequence[int]) -> Tensor:
    return PermuteDimsOp(x, list(axes)).outputs[0]


def concat(tensors: List[Tensor], axis: int) -> Tensor:
    if not isinstance(tensors, (list, tuple)) or any(not isinstance(t, Tensor) for t in tensors):
        raise ValueError('concat: expect a sequence of tensors, but got: {}'.format(type(tensors)))
    if any(tensors[0].dtype != t.dtype for t in tensors):
        raise ValueError(
            'concat: expect all tensors have the same dtype, but got:\n{}'.format(
                '\n'.join(t.signature() for t in tensors)
            )
        )
    return ConcatOp(*tensors, axis=axis).outputs[0]


def cast(x: Tensor, dtype: Union[str, DataType]) -> Tensor:
    dtype = data_type(dtype)
    if x.dtype == dtype:
        return x
    return CastOp(x, dtype).outputs[0]


def take(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    return TakeOp(data, indices, axis).outputs[0]


def gather(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    return GatherOp(data, indices, axis).outputs[0]


def index_select(data: Tensor, index: Tensor, dim: int) -> Tensor:
    return IdxSelOp(data, index, dim).outputs[0]


def repeat_interleave(data: Tensor, repeats: int, dim: int = None) -> Tensor:
    return RepeatInterleaveOp(data, repeats, dim=dim).outputs[0]


def strided_slice(
    data: Tensor,
    starts: Sequence[Optional[int]],
    ends: Sequence[Optional[int]],
    axes: Optional[Sequence[int]] = None,
    strides: Optional[Sequence[Optional[int]]] = None,
) -> Tensor:
    return StridedSliceOp(data, starts, ends, axes, strides).outputs[0]


def broadcast(data: Tensor, shape) -> Tensor:
    if same_shape(data.shape, shape):
        return data
    return BroadcastOp(data, shape).outputs[0]


def pad(data: Tensor, pads: List[int], mode: str = 'constant', value: float = 0.0) -> Tensor:
    if all(p == 0 for p in pads):
        return data
    return PadOp(data, pads, mode, value).outputs[0]


def conv_pad(data: Tensor, pads: Union[int, List[int]], value: float = 0.0) -> Tensor:
    from .utils import normalize_padding

    pads = normalize_padding(pads, dim=len(data.shape) - 2)
    return pad(data, pads, value=value)


def tile(data: Tensor, repeats: Sequence[int]) -> Tensor:
    """
    Tile a tensor. See https://numpy.org/doc/stable/reference/generated/numpy.tile.html.

    Parameters
    ----------
    data: Tensor
        The input tensor to be tiled.
    repeats: Sequence[int]
        A list of integers to represent the number of repeats for each dimension.
        Must have len(repeats) == len(data.shape).

    Returns
    -------
    ret: Tensor
        The tiled tensor, with shape [a * b for a, b in zip(data.shape, repeats)].
    """
    return TileOp(data, repeats).outputs[0]


def triu(x: Tensor, diagonal: int = 0) -> Tensor:
    rank = len(x.shape)
    if rank < 2:
        raise ValueError('triu operator expects a tensor with rank >= 2, but got {}'.format(x.shape))
    return TriuOp(x, diagonal).outputs[0]


def tril(x: Tensor, diagonal: int = 0) -> Tensor:
    rank = len(x.shape)
    if rank < 2:
        raise ValueError('triu operator expects a tensor with rank >= 2, but got {}'.format(x.shape))
    return TrilOp(x, diagonal).outputs[0]


def split(data: Tensor, parts_or_sections: Union[Sequence[int], int], axis: int = 0) -> List[Tensor]:
    if isinstance(parts_or_sections, int):
        if data.shape[axis] % parts_or_sections != 0:
            raise ValueError(
                'split operator expects the extent of given axis is divisible by number of sections, '
                'but got shape {}, axis {} and sections {}'.format(data.shape, axis, parts_or_sections)
            )
        parts = [data.shape[axis] // parts_or_sections] * parts_or_sections
    else:
        parts = parts_or_sections
        if sum(parts) != data.shape[axis]:
            raise ValueError(
                'split operator expects the sum(parts) parameter equals the the extent of given axis'
                ', but got shape {}, axis {} and parts {}'.format(data.shape, axis, parts)
            )

    axis = normalize_dim(axis, len(data.shape))
    outputs = []
    for i in range(len(parts)):
        start = sum(parts[:i])
        end = start + parts[i]
        outputs.append(strided_slice(data, starts=[start], ends=[end], axes=[axis], strides=[1]))
    return outputs


def expand_dims(x: Tensor, /, *, axis: int = 0) -> Tensor:
    axis = normalize_dim(axis, len(x.shape) + 1)
    new_shape = list(x.shape)
    new_shape.insert(axis, 1)
    return reshape(x, new_shape)


def meshgrid(*tensors: Tensor, indexing: str = "ij") -> List[Tensor]:
    if indexing not in ("xy", "ij"):
        raise ValueError(f"meshgrid: indexing must be 'xy' or 'ij', but got {indexing}")
    if isinstance(tensors[0], (list, tuple)):
        tensors = tensors[0]

    tensors = list(tensors)
    output_rank = len(tensors)
    if output_rank < 2:
        raise ValueError(f"meshgrid requires at least two tensors, but {output_rank} given.")

    if any(tensors[0].dtype != t.dtype for t in tensors):
        raise ValueError("meshgrid requires all inputs to be of the same type.")
    if any(not isinstance(t, Tensor) for t in tensors):
        raise ValueError("meshgrid: expect a sequence of tensors")

    # In torch.meshgrid, only 0D and 1D tensors are allowed.
    illegal_shapes = [t.shape for t in tensors if len(t.shape) > 1]
    if len(illegal_shapes) > 1:
        raise ValueError(
            f"meshgrid: only 0D and 1D tensors are allowed, but got a tensor with shape {illegal_shapes[0]}"
        )

    tensors_size = [t.size for t in tensors]

    outputs = []
    for dim in range(output_rank):
        target_shape = [1] * output_rank
        target_shape[dim] = -1
        grid = reshape(tensors[dim], target_shape)
        tile_shape = tensors_size[:]
        tile_shape[dim] = 1
        grid = tile(grid, tile_shape)
        if indexing == 'xy':
            grid = transpose(grid, (1, 0))
        outputs.append(grid)
    return outputs


def im2col(
    x: Tensor,
    kernel_size: Union[int, List[int]],
    dilation: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    stride: Union[int, List[Int]] = 1,
):
    nd = len(x.shape)
    if nd == 3:
        x = x.unsqueeze(0)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 2
    if isinstance(dilation, int):
        dilation = [dilation] * 2
    if isinstance(padding, int):
        padding = [padding] * 2
    if isinstance(stride, int):
        stride = [stride] * 2
    x = Im2ColOp(x, kernel_size, dilation, padding, stride).outputs[0]

    if nd == 3:
        return x.squeeze(0)
    return x


def as_strided(
    x: Tensor, size: Union[int, List[int]], stride: Union[int, List[int]], storage_offset: Optional[int] = None
):
    return AsStridedOp(x, size, stride, storage_offset).outputs[0]


def flip(x: Tensor, dims: Union[List[int], Tuple[int]]):
    return FlipOp(x, dims).outputs[0]
