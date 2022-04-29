from typing import List, Optional, Union, Sequence

from hidet.ir.expr import And, if_then_else, convert
from hidet.ir.layout import DataLayout, RowMajorLayout, ColumnMajorLayout
from hidet.utils import prod
from .utils import Task, InverseMap, Operator, Tensor, TensorNode, compute, input_like, normalize_dim


def same_shape(shape_a: List[int], shape_b: List[int]) -> bool:
    return len(shape_a) == len(shape_b) and all(a == b for a, b in zip(shape_a, shape_b))


class ReshapeTask(Task):
    def __init__(self, x: TensorNode, shape: List[int]):
        assert prod(x.const_shape()) == prod(shape)
        x_layout = x.data_type.layout
        y_layout = DataLayout.row_major(shape)

        def index_map(y_indices):
            idx = y_layout(*y_indices)
            cur = 1
            x_indices = []
            for dim in reversed(x_layout.shape):
                x_indices.append((idx // cur) % dim)
                cur = cur * dim
            return list(reversed(x_indices))

        y = compute(
            name='y',
            shape=y_layout.shape,
            fcompute=lambda *indices: x[index_map(indices)],
            scope='global',
            layout=y_layout
        )
        super().__init__(
            name='reshape',
            inputs=[x],
            outputs=[y]
        )


class RearrangeTask(Task):
    def __init__(self, x: TensorNode, plan: List[List[int]]):
        x_shape = x.const_shape()
        y_shape = [prod([x_shape[i] for i in dims]) for dims in plan]

        def index_split(total_index, dim_sizes: List[int]) -> List:
            bases = [prod(dim_sizes[i + 1:]) for i in range(len(dim_sizes))]
            return [(total_index // base) % dim for dim, base in zip(dim_sizes, bases)]

        def fcompute(*y_indices):
            x_indices = [None for _ in range(len(x_shape))]
            for i, y_index in enumerate(y_indices):
                dims = plan[i]
                if len(dims) == 0:
                    # this new dimension has size 1
                    continue
                else:
                    split_indices = index_split(total_index=y_index, dim_sizes=[x_shape[k] for k in dims])
                    for j, x_index in zip(dims, split_indices):
                        x_indices[j] = x_index
            for i, x_index in enumerate(x_indices):
                if x_index is None:
                    if x_shape[i] != 1:
                        msg = 'Rearrange plan {} on tensor {} leave non-one dimension {} not been accessed'.format(plan, x_shape, i)
                        raise ValueError(msg)
                    else:
                        x_indices[i] = 0
            return x[x_indices]

        y = compute('y', y_shape, fcompute, scope='global')

        def inverse_map(*x_indices):
            y_indices = []
            for dims in plan:
                cnt = convert(0)
                for dim in dims:
                    cnt = cnt * x_shape[dim] + x_indices[dim]
                y_indices.append(cnt)
            return y_indices

        super().__init__(
            name='rearrange',
            inputs=[x],
            outputs=[y],
            inverse_map={x: InverseMap.from_lambda(inverse_map, len(x_shape))}
        )


class ConcatTask(Task):
    def __init__(self, inputs: List[TensorNode], axis: int):
        shapes = [t.const_shape() for t in inputs]
        n = len(shapes)
        assert n > 0
        for i in range(1, n):
            if len(shapes[0]) != len(shapes[i]):
                raise ValueError('Concat: all shapes must have the same rank, got {}'.format(shapes))
            if any(a != b for j, (a, b) in enumerate(zip(shapes[0], shapes[i])) if j != axis):
                raise ValueError('Concat: all tensors must have the same shape except axis dimension, got {}, axis {}'.format(shapes, axis))
        rank = len(shapes[0])
        out_shape = [shapes[0][i] if i != axis else sum(shapes[j][i] for j in range(n)) for i in range(rank)]

        def fmap(*indices):
            pre_sum = [sum([shapes[j][axis] for j in range(i)]) for i in range(n + 1)]
            value = inputs[-1][indices[:axis] + (indices[axis] - pre_sum[-2],) + indices[axis+1:]]
            for i, input in reversed(list(zip(range(n-1), inputs[:n-1]))):
                input_i_value = inputs[i][indices[:axis] + (indices[axis] - pre_sum[i],) + indices[axis+1:]]
                value = if_then_else(indices[axis] < pre_sum[i + 1], input_i_value, value)
            return value

        out = compute(
            name='out',
            shape=out_shape,
            fcompute=lambda *indices: fmap(*indices)
        )

        super().__init__(
            name='concat',
            inputs=inputs,
            outputs=[out]
        )


class TakeTask(Task):
    def __init__(self, data: TensorNode, indices: TensorNode, axis=0):
        data_shape = data.const_shape()
        indices_shape = indices.const_shape()
        output_shape = data_shape[:axis] + indices_shape + data_shape[axis + 1:]
        assert 0 <= axis < len(data_shape)

        def fmap(*output_indices):
            indices_indices = output_indices[axis: axis + len(indices_shape)]
            data_indices = output_indices[:axis] + (indices[indices_indices],) + output_indices[axis + len(indices_shape):]
            return data[data_indices]

        output = compute(
            name='output',
            shape=output_shape,
            fcompute=lambda *output_indices: fmap(*output_indices),
            scope='global'
        )
        super().__init__(
            name='take',
            inputs=[data, indices],
            outputs=[output]
        )


class StridedSliceTask(Task):
    def __init__(self, data: TensorNode, starts: List[Optional[int]], ends: List[Optional[int]], axes: List[int], strides: List[int]):
        assert len(starts) == len(ends) == len(axes) == len(strides)
        if len(axes) != len(set(axes)):
            raise ValueError('Duplicated axes in slice, axes: {}'.format(axes))
        data_shape = data.const_shape()
        output_shape = list(data_shape)
        axis2info = {}
        for axis, start, end, stride in zip(axes, starts, ends, strides):
            if stride == 0:
                raise NotImplementedError('Stride can not be 0 in slicing: starts {} ends {} axes {} strides {}.'.format(starts, ends, axes, strides))
            if stride > 0:
                output_shape[axis] = (end - start + stride - 1) // stride
            else:
                output_shape[axis] = (start - end + (-stride) - 1) // (-stride)
            if output_shape[axis] <= 0:
                raise NotImplementedError('Slice result can not be: starts {} ends {} axes {} strides {}'.format(starts, ends, axes, strides))
            axis2info[axis] = (start, end, stride)

        def fmap(indices):
            data_indices = []
            for axis, index in enumerate(indices):
                if axis in axis2info:
                    start, end, stride = axis2info[axis]
                    data_indices.append(start + index * stride)
                else:
                    data_indices.append(index)
            return data[data_indices]

        out = compute(
            'out',
            shape=output_shape,
            fcompute=lambda *indices: fmap(indices),
            scope=data.data_type.scope
        )
        super().__init__(
            name='slice',
            inputs=[data],
            outputs=[out]
        )


def can_broadcast(src_shape: List[int], dst_shape: List[int]) -> bool:
    if len(dst_shape) < len(src_shape):
        return False
    src_shape = [1 for _ in range(len(dst_shape) - len(src_shape))] + src_shape
    for a, b in zip(src_shape, dst_shape):
        if a != 1 and a != b:
            return False
    return True


class BroadcastTask(Task):
    def __init__(self, data: TensorNode, shape: List[int]):
        data_shape = data.const_shape()
        if not can_broadcast(data_shape, shape):
            raise ValueError('Can not broadcast a tensor with shape {} to {}'.format(data_shape, shape))

        def fmap(*indices):
            expanded = len(shape) - len(data_shape)
            indices = indices[expanded:]
            indices = [v if data_shape[i] != 1 else 0 for i, v in enumerate(indices)]
            return data[indices]

        out = compute(
            'out',
            shape=shape,
            fcompute=fmap,
            scope=data.data_type.scope
        )
        super().__init__(
            name='broadcast',
            inputs=[data],
            outputs=[out]
        )


class PadTask(Task):
    def __init__(self, data: TensorNode, pads: List[int], value: float):
        shape = data.const_shape()
        rank = len(shape)
        assert rank * 2 == len(pads)
        out_shape = [a + b + c for a, b, c in zip(pads[:rank], shape, pads[rank:])]

        def fmap(*indices):
            indices = [idx - beg for idx, beg in zip(indices, pads[:rank])]
            cond = And.join_list([And(0 <= idx, idx < shape[i]) for i, idx in enumerate(indices)])
            return if_then_else(cond, data[indices], value)

        out = compute(
            'out',
            shape=out_shape,
            fcompute=fmap,
            scope=data.data_type.scope
        )
        super().__init__(
            name='pad',
            inputs=[data],
            outputs=[out]
        )


class ReshapeOp(Operator):
    def __init__(self, x: Tensor, shape):
        shape = self.normalize_shape(x.shape, shape)
        task = ReshapeTask(input_like(x, 'x'), shape)
        super().__init__(
            inputs=[x],
            task=task,
            shape=shape
        )

    @staticmethod
    def normalize_shape(origin_shape: List[int], shape: List[int]):
        # [1, 3, 224, 224], [1, -1, 224, 0] => [1, 3, 224, 224]
        shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == 0:
                if i >= len(origin_shape):
                    raise ValueError('0 is used outside original shape: origin {} target {}'.format(origin_shape, shape))
                shape[i] = origin_shape[i]
        size = prod(origin_shape)
        cnt = sum([1 for v in shape if v == -1])
        if cnt == 0:
            if prod(shape) != size:
                raise ValueError('Reshape: given shape has different size with input tensor: shape {} and size {}'.format(shape, size))
            return shape
        elif cnt == 1:
            remain_size = prod([v for v in shape if v != -1])
            if size % remain_size != 0:
                raise ValueError('Given shape is incompatible with input tensor: shape {} and size {}'.format(shape, size))
            return [v if v != -1 else size // remain_size for v in shape]
        else:
            raise ValueError('Can not infer the shape when there are multiple -1: {}'.format(shape))


class RearrangeOp(Operator):
    def __init__(self, x: Tensor, plan: List[List[int]]):
        super().__init__(
            inputs=[x],
            task=RearrangeTask(input_like(x, 'x'), plan=plan),
            plan=plan
        )


class SqueezeOp(Operator):
    def __init__(self, x: Tensor, dims: List[int]):
        super().__init__(
            inputs=[x],
            task=RearrangeTask(input_like(x, 'x'), plan=[[i] for i in range(len(x.shape)) if i not in dims]),
            dims=dims
        )

    def imperative_run(self, inputs: Optional[List[Tensor]] = None) -> List[Tensor]:
        x = inputs[0] if inputs else self.inputs[0]
        if isinstance(x.layout, (RowMajorLayout, ColumnMajorLayout)):
            shape = self.task.params[1].const_shape()
            layout = x.layout.__class__(shape)
            return [Tensor(shape=shape, dtype=x.dtype, device=x.device, storage=x.storage, layout=layout, trace=None)]
        else:
            return Operator.imperative_run(self, inputs)


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
        assert c == len(x.shape)
        super().__init__(
            inputs=[x],
            task=RearrangeTask(input_like(x, 'x'), plan=plan),
            dims=dims
        )

    def imperative_run(self, inputs: Optional[List[Tensor]] = None) -> List[Tensor]:
        x = inputs[0] if inputs else self.inputs[0]
        if isinstance(x.layout, (RowMajorLayout, ColumnMajorLayout)):
            shape = self.task.outputs[0].const_shape()
            layout = x.layout.__class__(shape)
            return [Tensor(shape=shape, dtype=x.dtype, device=x.device, storage=x.storage, layout=layout, trace=None)]
        else:
            return Operator.imperative_run(self, inputs)


class FlattenOp(Operator):
    def __init__(self, x: Tensor, start_dim: int, end_dim: int):
        rank = len(x.shape)
        start_dim = normalize_dim(start_dim, rank)
        end_dim = normalize_dim(end_dim, rank)
        assert 0 <= start_dim < end_dim <= rank
        dims = list(range(len(x.shape)))
        plan = [[v] for v in dims[:start_dim]] + [dims[start_dim: end_dim]] + [[v] for v in dims[end_dim:]]
        super().__init__(
            inputs=[x],
            task=RearrangeTask(input_like(x, 'x'), plan=plan),
            start_dim=start_dim,
            end_dim=end_dim
        )


class TransposeOp(Operator):
    def __init__(self, x: Tensor, axes: Optional[List[int]] = None):
        if axes and len(axes) != len(x.shape):
            raise ValueError('Transpose tensor with shape {} expect a permutation of axes with length {}, got {}'.format(x.shape, len(x.shape), axes))
        if axes is None:
            axes = list(reversed(range(len(x.shape))))
        plan = [[v] for v in axes]
        super().__init__(
            inputs=[x],
            task=RearrangeTask(input_like(x, 'x'), plan),
            axes=axes
        )


class CastOp(Operator):
    def __init__(self, x: Tensor, dtype: str):
        from hidet.ir.expr import Cast
        from .arithmatic import UnaryElementwiseTask
        super().__init__(
            inputs=[x],
            task=UnaryElementwiseTask('cast', input_like(x, 'x'), op=lambda v: Cast(v, dtype)),
            dtype=dtype
        )


class ConcatOp(Operator):
    def __init__(self, tensors: List[Tensor], axis: int):
        if len(tensors) == 0:
            raise ValueError('Concat requires at least one tensor, 0 given.')
        axis = normalize_dim(axis, len(tensors[0].shape))
        super().__init__(
            inputs=tensors,
            task=ConcatTask([input_like(tensor, 'x{}'.format(idx)) for idx, tensor in enumerate(tensors)], axis=axis),
            axis=axis
        )


class TakeOp(Operator):
    def __init__(self, data: Tensor, indices: Tensor, axis: int):
        super().__init__(
            inputs=[data, indices],
            task=TakeTask(input_like(data, 'data'), input_like(indices, 'indices'), axis=axis),
            axis=axis
        )


class StridedSliceOp(Operator):
    def __init__(self, data: Tensor, starts: List[int], ends: List[int], axes: Optional[List[int]] = None, strides: Optional[List[int]] = None):
        starts, ends, axes, strides = self.normalize(data.shape, starts, ends, axes, strides)
        task = StridedSliceTask(input_like(data, 'data'), starts, ends, axes, strides)
        super().__init__(
            inputs=[data],
            task=task,
            starts=starts,
            ends=ends,
            axes=axes,
            strides=strides
        )

    @staticmethod
    def normalize(shape, starts, ends, axes: Optional, strides: Optional):
        # follow: https://github.com/onnx/onnx/blob/main/docs/Operators.md#slice to normalize
        rank = len(shape)
        if axes is None:
            axes = [i for i in range(len(starts))]
        axes = normalize_dim(axes, rank)
        if strides is None:
            strides = [1 for _ in range(len(starts))]
        shape = [shape[i] for i in axes]
        assert len(shape) == len(starts) == len(ends) == len(axes) == len(strides)
        for i in range(len(axes)):
            starts[i] = starts[i] + shape[i] if starts[i] < 0 else starts[i]
            ends[i] = ends[i] + shape[i] if ends[i] < 0 else ends[i]
            if strides[i] > 0:
                starts[i] = max(0, min(shape[i], starts[i]))
                ends[i] = max(0, min(shape[i], ends[i]))
            else:
                starts[i] = max(0, min(shape[i] - 1, starts[i]))
                ends[i] = max(-1, min(shape[i] - 1, ends[i]))
        return starts, ends, axes, strides


class BroadcastOp(Operator):
    def __init__(self, data: Tensor, shape: List[int]):
        super().__init__(
            inputs=[data],
            task=BroadcastTask(input_like(data, 'data'), shape),
            shape=shape
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
            task=PadTask(input_like(data, 'data'), pads, value),
            pads=pads,
            mode=mode,
            value=value
        )


def reshape(x: Tensor, shape) -> Tensor:
    if same_shape(x.shape, shape):
        return x
    return ReshapeOp(x, shape).get_output(0)


def rearrange(x: Tensor, plan: List[List[int]]) -> Tensor:
    """
    Rearrange a tensor. This task is a general task of squeeze, unsqueeze, flatten, and perm.

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
    squeeze([1, 1, 2, 3], dims=[0, 1]) = rearrange([1, 1, 2, 3], plan=[[2], [3]]) => Tensor([2, 3])
    unsqueeze([2, 3], dims=[0, 1]) = rearrange([2, 3], plan=[[], [], [0], [1]]) => Tensor([1, 1, 2, 3])
    flatten([2, 3, 4, 5], start_dim=1, end_dim=2) = rearrange([2, 3, 4, 5], plan=[[0], [1, 2], [3]]) => Tensor([2, 12, 5])
    """
    return RearrangeOp(x, plan).get_output(0)


def squeeze(x: Tensor, dims: Union[int, Sequence[int]]) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return SqueezeOp(x, dims).get_output(0)


def unsqueeze(x: Tensor, dims: Union[int, Sequence[int]]) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return UnsqueezeOp(x, dims).get_output(0)


def flatten(x: Tensor, start_dim=0, end_dim=None) -> Tensor:
    start_dim = normalize_dim(start_dim, len(x.shape))
    end_dim = normalize_dim(end_dim, len(x.shape))
    if start_dim + 1 == end_dim:
        return x
    return FlattenOp(x, start_dim, end_dim).get_output(0)


def transpose(x: Tensor, axes: Optional[List[int]]) -> Tensor:
    return TransposeOp(x, axes).get_output(0)


def concat(tensors: List[Tensor], axis: int) -> Tensor:
    return ConcatOp(tensors, axis).get_output(0)


def cast(x: Tensor, dtype: str) -> Tensor:
    return CastOp(x, dtype).get_output(0)


def take(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    return TakeOp(data, indices, axis).get_output(0)


def strided_slice(data: Tensor, starts: List[int], ends: List[int], axes: Optional[List[int]] = None, strides: Optional[List[int]] = None) -> Tensor:
    return StridedSliceOp(data, starts, ends, axes, strides).get_output(0)


def broadcast(data: Tensor, shape) -> Tensor:
    if same_shape(data.shape, shape):
        return data
    return BroadcastOp(data, shape).get_output(0)


def pad(data: Tensor, pads: List[int], mode: str = 'constant', value: float = 0.0) -> Tensor:
    return PadOp(data, pads, mode, value).get_output(0)
