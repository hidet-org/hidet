from typing import Tuple, List

from .base import register_task_layout, TaskLayout, Int, task_layout_compose
from .generic import full_layout, RowMajorLayout, ColumnMajorLayout


class WarpLayout4x8(TaskLayout):
    def __init__(self):
        def worker2task(worker_index: Int) -> List[Tuple[Int, ...]]:
            ti = worker_index % 2 + (worker_index // 16) * 2
            tj = (worker_index % 16) // 2
            return [(ti, tj)]

        def task2worker(task_index: Tuple[Int, ...]) -> Int:
            i, j = task_index
            return i // 2 * 16 + j * 2 + i % 2

        super().__init__(32, (4, 8), worker2task, task2worker)


# workers = 32
for in_m in [1, 4]:
    for in_n in [1, 4]:
        for out_n in [1, 2]:
            for out_m in [1, 2]:
                register_task_layout((full_layout(out_n, out_m) * WarpLayout4x8()) * full_layout(in_m, in_n))

# workers in [32, 64, ..., 1024], rank = 2
for num_warps in range(1, 32 + 1):
    row_major_layouts = list(RowMajorLayout.get_layouts(num_workers=num_warps * 32, rank=2))
    col_major_layouts = list(ColumnMajorLayout.get_layouts(num_workers=num_warps * 32, rank=2))
    for row_major_layout in row_major_layouts:
        for in_m in [1, 4]:
            for in_n in [1, 4]:
                for out_n in [1, 2]:
                    for out_m in [1, 2]:
                        register_task_layout((full_layout(out_n, out_m) * row_major_layout) * full_layout(in_m, in_n))
    for col_major_layout in col_major_layouts:
        for in_m in [1, 4]:
            for in_n in [1, 4]:
                for out_n in [1, 2]:
                    for out_m in [1, 2]:
                        register_task_layout((full_layout(out_n, out_m) * col_major_layout) * full_layout(in_m, in_n))

