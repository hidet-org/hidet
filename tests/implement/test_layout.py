import pytest
from itertools import product

from hidet.ir.layout import get_task_layouts, TaskLayout
from hidet.ir.layout.generic import RowMajorLayout, ColumnMajorLayout


def check_layout(layout):
    num_workers = layout.num_workers
    task_shape = layout.task_shape
    worker2tasks = {worker: layout.worker2task(worker) for worker in range(num_workers)}
    # check task2thread
    for task_index in product(*[range(s) for s in task_shape]):
        worker_index = layout.task2worker(task_index)
        assert tuple(task_index) in worker2tasks[worker_index]
    # check thread2task
    for worker_index in range(num_workers):
        for worker_task in worker2tasks[worker_index]:
            expect_worker = layout.task2worker(worker_task)
            assert expect_worker == worker_index


@pytest.mark.parametrize("layout_generator, num_workers, task_shape, rank, expected_layouts",
                         [
                             (RowMajorLayout(), 32, (4, 8), 2, 1),
                             (RowMajorLayout(), 32, None, 2, 6),  # (1, 32) (2, 16) (4, 8) (8, 4) (16, 2) (32, 1)
                             (RowMajorLayout(), None, (4, 8), 2, 1),
                             (ColumnMajorLayout(), 32, (4, 8), 2, 1),
                             (ColumnMajorLayout(), 32, None, 2, 6),  # (1, 32) (2, 16) (4, 8) (8, 4) (16, 2) (32, 1)
                             (ColumnMajorLayout(), None, (4, 8), 2, 1),
                         ])
def test_layout_generators(layout_generator, num_workers, task_shape, rank, expected_layouts):
    layouts = layout_generator.get_layouts(num_workers, task_shape, rank)
    assert len(list(layouts)) == expected_layouts
    for layout in layouts:
        check_layout(layout)


@pytest.mark.skip(reason='Take too long time')
def test_layouts():
    for idx, layout in enumerate(TaskLayout.registered):
        check_layout(layout)


if __name__ == '__main__':
    pytest.main(__file__)
