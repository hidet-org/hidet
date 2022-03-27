from hidet.ir.dialects.compute import tensor_input, compute
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
        src = tensor_input('src', 'float', shape=src_layout.shape)
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



