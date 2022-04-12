from typing import List
from hidet.ir.dialects.compute import tensor_input, compute, reduce, custom_compute
from hidet.ir.layout.data_layout import DataLayout, RowMajorLayout, ColumnMajorLayout
from hidet.ir.type import tensor_type
from hidet.ir.expr import AlterLayout, Cast
from hidet.ir.task import Task, Grid
from hidet.utils import prod


def copy(src_layout: DataLayout, dst_layout: DataLayout) -> Task:
    assert prod(src_layout.shape) == prod(dst_layout.shape)
    if isinstance(src_layout, RowMajorLayout) and isinstance(dst_layout, RowMajorLayout):
        compatible = True
    elif isinstance(src_layout, ColumnMajorLayout) and isinstance(dst_layout, ColumnMajorLayout):
        compatible = True
    else:
        compatible = False
    if compatible:
        def layout_map(*dst_indices):
            src_indices = []
            global_index = dst_layout(*dst_indices)
            src_shape = src_layout.shape
            cur = 1
            for dim in src_shape:
                src_indices.append(global_index // cur % dim)
                cur = cur * dim
            return src_indices

        src = tensor_input('src', 'float32', shape=src_layout.shape, scope='global', layout=src_layout)
        dst = compute(
            name='dst',
            shape=dst_layout.shape,
            fcompute=lambda *indices: AlterLayout(src, shape=dst_layout.shape, layout_map=layout_map)[indices],
            scope='global',
            layout=dst_layout
        )
        return Task(
            name='copy',
            computation=dst,
            params=[src, dst],
            worker=Grid()
        )
    else:
        def index_map(dst_indices):
            idx = dst_layout(*dst_indices)
            cur = 1
            src_indices = []
            for dim in reversed(src_layout.shape):
                src_indices.append((idx // cur) % dim)
                cur = cur * dim
            return list(reversed(src_indices))

        src = tensor_input('src', 'float32', shape=src_layout.shape, scope='global', layout=src_layout)
        dst = compute(
            name='dst',
            shape=dst_layout.shape,
            fcompute=lambda *indices: src[index_map(indices)],
            scope='global',
            layout=dst_layout
        )
        return Task(
            name='copy',
            computation=dst,
            params=[src, dst],
            worker=Grid()
        )


def reduce_mean(x_layout: DataLayout, dims: List[int], keep_dim=False):
    x_shape = [int(v) for v in x_layout.shape]
    y_shape = []
    for i in range(len(x_shape)):
        if i in dims:
            if keep_dim:
                y_shape.append(1)
        else:
            y_shape.append(x_shape[i])
    x = tensor_input('x', 'float32', x_shape, 'global', x_layout)

    def fcompute(*indices):
        def reduce_fcompute(*reduce_indices):
            x_indices = []
            p = 0
            q = 0
            for i in range(len(x_shape)):
                if i not in dims:
                    x_indices.append(indices[p])
                    p += 1
                else:
                    x_indices.append(reduce_indices[q])
                    q += 1
                    if keep_dim:
                        p += 1
            assert p == len(indices) and q == len(reduce_indices)
            return x[x_indices]

        reduce_shape = [x_shape[i] for i in dims]
        return reduce(shape=reduce_shape, fcompute=reduce_fcompute, reduce_type='avg')

    y = compute(name='y', shape=y_shape, fcompute=fcompute, scope='global')
    return Task(
        'reduce_mean',
        computation=y,
        params=[x, y],
        worker=Grid()
    )


def squeeze(x_layout: DataLayout, dims) -> Task:
    x_shape = [int(v) for v in x_layout.shape]
    y_shape = []
    for i in range(len(x_shape)):
        if i in dims:
            assert x_shape[i] == 1
        else:
            y_shape.append(x_shape[i])
    x = tensor_input('x', 'float32', shape=x_shape, scope='global', layout=x_layout)

    def fcompute(*y_indices):
        x_indices = []
        p = 0
        for i in range(len(x_shape)):
            if i in dims:
                x_indices.append(0)
            else:
                x_indices.append(y_indices[p])
                p += 1
        return x[x_indices]

    y = compute('y', y_shape, fcompute, scope='global')
    return Task(
        'squeeze',
        computation=y,
        params=[x, y],
        worker=Grid()
    )


def unsqueeze(x_layout: DataLayout, dims) -> Task:
    x_shape = [int(v) for v in x_layout.shape]
    y_shape = []
    p = 0
    for i in range(len(x_shape) + len(dims)):
        if i in dims:
            y_shape.append(1)
        else:
            assert p < len(x_shape)
            y_shape.append(x_shape[p])
            p += 1
    assert p == len(x_shape)
    x = tensor_input('x', 'float32', x_shape, scope='global', layout=x_layout)

    def fcompute(*y_indices):
        x_indices = [axis for i, axis in enumerate(y_indices) if i not in dims]
        return x[x_indices]

    y = compute('y', y_shape, fcompute, scope='global')
    return Task(
        'unsqueeze',
        computation=y,
        params=[x, y],
        worker=Grid()
    )


def index_split(total_index, dim_sizes: List[int]) -> List:
    bases = [prod(dim_sizes[i + 1:]) for i in range(len(dim_sizes))]
    return [(total_index // base) % dim for dim, base in zip(dim_sizes, bases)]


def rearrange(x_layout: DataLayout, plan: List[List[int]]) -> Task:
    """
    Rearrange a tensor. This task is a general task of squeeze, unsqueeze, flatten, and perm.

    Parameters
    ----------
    x_layout: DataLayout
        The data layout of input.

    plan: List[List[int]]
        The rearrange plan.

    Returns
    -------
    ret: Task
        The task to conduct rearrangement.

    Examples
    --------
    squeeze([1, 1, 2, 3], dims=[0, 1]) = rearrange([1, 1, 2, 3], plan=[[2], [3]]) => Tensor([2, 3])
    unsqueeze([2, 3], dims=[0, 1]) = rearrange([2, 3], plan=[[], [], [0], [1]]) => Tensor([1, 1, 2, 3])
    flatten([2, 3, 4, 5], start_dim=1, end_dim=2) = rearrange([2, 3, 4, 5], plan=[[0], [1, 2], [3]]) => Tensor([2, 12, 5])
    """
    x_shape = [int(v) for v in x_layout.shape]
    y_shape = [prod([x_shape[i] for i in dims]) for dims in plan]
    x = tensor_input('x', 'float32', x_shape, 'global', x_layout)

    def fcompute(*y_indices):
        x_indices = [None for _ in range(len(x_shape))]
        for i, y_index in enumerate(y_indices):
            dims = plan[i]
            if len(dims) == 0:
                # this dimension has size 1
                continue
            else:
                split_indices = index_split(total_index=y_index, dim_sizes=[x_shape[k] for k in dims])
                for j, x_index in zip(dims, split_indices):
                    x_indices[j] = x_index
        assert all(x_index is not None for x_index in x_indices)
        return x[x_indices]

    y = compute('y', y_shape, fcompute, scope='global')
    return Task(
        'rearrange',
        computation=y,
        params=[x, y],
        worker=Grid()
    )


def cast(x_layout: DataLayout, src_dtype: str, dst_dtype: str) -> Task:
    shape = [int(v) for v in x_layout.shape]
    x = tensor_input('x', src_dtype, shape, layout=x_layout)
    y = compute('y', shape, lambda *indices: Cast(x[indices], dst_dtype))
    return Task(
        'cast',
        computation=y,
        params=[x, y],
        worker=Grid()
    )


def concat(layouts: List[DataLayout], axis: int) -> Task:
    shapes = [[int(v) for v in layout.shape] for layout in layouts]
    n = len(shapes)
    assert len(shapes) > 0
    for i in range(1, n):
        assert len(shapes[0]) == len(shapes[i]), 'all shapes must have the same rank'
        assert all(a == b for j, (a, b) in enumerate(zip(shapes[0], shapes[i])) if j != axis), 'all tensors must have the same shape except axis dimension'
    rank = len(shapes[0])
    out_shape = [shapes[0][i] if i != axis else sum(shapes[j][i] for j in range(n)) for i in range(rank)]
    input_params = [tensor_input('x{}'.format(i), 'float32', shape, scope='global', layout=layout) for i, (shape, layout) in enumerate(zip(shapes, layouts))]
    params = input_params + [tensor_input('out', 'float32', out_shape)]
    out = custom_compute('concat', tensor_type('global', 'float32', shape=out_shape)),
    return Task(
        name='concat',
        computation=out,
        params=params,
        worker=Grid()
    )


def take(data_layout: DataLayout, indices_layout: DataLayout, axis=0) -> Task:
    data_shape = data_layout.const_shape()
    indices_shape = indices_layout.const_shape()
    output_shape = data_shape[:axis] + indices_shape + data_shape[axis + 1:]
    assert 0 <= axis < len(data_shape)

    data = tensor_input('data', 'float32', data_shape, 'global', data_layout)
    indices_tensor = tensor_input('indices', 'int64', indices_shape, 'global', indices_layout)

    def fmap(*indices):
        indices_indices = indices[axis: axis + len(indices_shape)]
        data_indices = indices[:axis] + (indices_tensor[indices_indices],) + indices[axis + len(indices_shape):]
        return data[data_indices]

    output = compute(
        name='output',
        shape=output_shape,
        fcompute=lambda *indices: fmap,
        scope='global'
    )
    return Task(
        name='take',
        computation=output,
        params=[data, indices_tensor, output],
        worker=Grid()
    )


def strided_slice(data_layout: DataLayout, starts: List[int], ends: List[int], axes: List[int] = None, steps: List[int] = None) -> Task:
    data_shape = data_layout.const_shape()
    data_rank = len(data_shape)
    assert len(starts) == len(ends)
    rank = len(starts)
    axes = axes if axes else list(range(rank))
    steps = steps if steps else [1 for _ in range(rank)]
