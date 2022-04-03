from typing import List
from hidet.ir.dialects.compute import tensor_input, compute, reduce, custom_compute
from hidet.ir.layout.data_layout import DataLayout, RowMajorLayout, ColumnMajorLayout
from hidet.ir.type import tensor_type
from hidet.ir.expr import AlterLayout
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

        src = tensor_input('src', 'float32', shape=src_layout.shape)
        dst = compute(
            name='dst',
            shape=dst_layout.shape,
            fcompute=lambda *indices: AlterLayout(src, shape=dst_layout.shape, layout_map=layout_map)[indices]
        )
        return Task(
            name='copy',
            computation=dst,
            params=[src, dst],
            params_type=[
                tensor_type('global', 'float32', layout=src_layout),
                tensor_type('global', 'float32', layout=dst_layout)
            ],
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

        src = tensor_input('src', 'float32', shape=src_layout.shape)
        dst = compute(
            name='dst',
            shape=dst_layout.shape,
            fcompute=lambda *indices: src[index_map(indices)],
        )
        return Task(
            name='copy',
            computation=dst,
            params=[src, dst],
            params_type=[
                tensor_type('global', 'float32', layout=src_layout),
                tensor_type('global', 'float32', layout=dst_layout),
            ],
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
    x = tensor_input('x', 'float32', x_shape)

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

    y = compute(name='y', shape=y_shape, fcompute=fcompute)
    return Task(
        'reduce_mean',
        computation=y,
        params=[x, y],
        params_type=[
            tensor_type('global', 'float32', layout=x_layout),
            tensor_type('global', 'float32', shape=y_shape)
        ],
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
    x = tensor_input('x', 'float32', shape=x_shape)

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

    y = compute('y', y_shape, fcompute)
    return Task(
        'squeeze',
        computation=y,
        params=[x, y],
        params_type=[
            tensor_type('global', 'float32', layout=x_layout),
            tensor_type('global', 'float32', shape=y_shape)
        ],
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
    x = tensor_input('x', 'float32', x_shape)

    def fcompute(*y_indices):
        x_indices = [axis for i, axis in enumerate(y_indices) if i not in dims]
        return x[x_indices]

    y = compute('y', y_shape, fcompute)
    return Task(
        'unsqueeze',
        computation=y,
        params=[x, y],
        params_type=[
            tensor_type('global', 'float32', layout=x_layout),
            tensor_type('global', 'float32', shape=y_shape)
        ],
        worker=Grid()
    )


def concat(layouts: List[DataLayout], axis: int):
    shapes = [[int(v) for v in layout.shape] for layout in layouts]
    n = len(shapes)
    assert len(shapes) > 0
    for i in range(1, n):
        assert len(shapes[0]) == len(shapes[i]), 'all shapes must have the same rank'
        assert all(a == b for j, (a, b) in enumerate(zip(shapes[0], shapes[i])) if j != axis), 'all tensors must have the same shape except axis dimension'
    rank = len(shapes[0])
    out_shape = [shapes[0][i] if i != axis else sum(shapes[j][i] for j in range(n)) for i in range(rank)]
    input_params = [tensor_input('x{}'.format(i), 'float32', shape) for i, shape in enumerate(shapes)]
    input_params_type = [tensor_type('global', 'float32', layout=layout) for layout in layouts]
    params = input_params + [tensor_input('out', 'float32', out_shape)]
    params_type = input_params_type + [tensor_type('global', 'float32', shape=out_shape)]
    return Task(
        name='concat',
        computation=custom_compute('concat'),
        params=params,
        params_type=params_type,
        worker=Grid()
    )
