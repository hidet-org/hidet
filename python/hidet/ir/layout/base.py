from typing import Union, Tuple, List, Type, Dict, Optional, Sequence, Iterator, Callable, Iterable
from hidet.ir.expr import Constant
from hidet.ir.expr import Expr, var

Int = Union[Expr, int]


class DataLayout:
    pass


class StridesLayout(DataLayout):
    def __init__(self, strides):
        super().__init__()
        self.strides: List[Int] = strides
        self.global2serial: Callable[[Int, ...], Int] = (lambda *indices: sum(v * self.strides[i] for i, v in enumerate(indices)))


class LocalLayout(DataLayout):
    def __init__(self):
        super().__init__()
        self.local_size = None
        self.local2global: Callable[[Int, Int], Sequence[Int]] = None
        self.global2local: Callable[[Int, ...], Tuple[Int, Int]] = None


class TaskLayout:
    registered = []

    def __init__(self,
                 num_workers: int = None,
                 task_shape: Tuple[int, ...] = None,
                 worker2task: Optional[Callable[[Int], List[Tuple[Int, ...]]]] = None,
                 task2worker: Optional[Callable[[Tuple[Int, ...]], Int]] = None):
        self.num_workers: int = num_workers
        self.task_shape: Tuple[int, ...] = task_shape
        self.worker2task: Callable[[Int], List[Sequence[Int, ...]]] = worker2task
        self.task2worker: Callable[[Sequence[Int, ...]], Int] = task2worker

    def __mul__(self, other):
        assert isinstance(other, TaskLayout)
        return task_layout_compose(self, other)

    def projection(self, *values: List[Optional[int]]) -> 'TaskLayout':
        assert len(values) == len(self.task_shape)

        def task2worker(*args):
            raise NotImplementedError()

        def worker2task(w: Int):
            rank = len(self.task_shape)
            projected_tasks = []
            for task in self.worker2task(w):
                projected_tasks.append(tuple(task[i] if values[i] is None else values[i] for i in range(rank)))
            return projected_tasks
        return TaskLayout(self.num_workers, self.task_shape, worker2task, task2worker)

    @property
    def expr_text(self):
        from hidet.ir.functors.simplifier import simplify, convert
        rank = len(self.task_shape)
        assert rank <= 5
        indices = ['i', 'j', 'k', 'p', 'q']
        vars = [var(name) for name in indices[:rank]]
        return str(simplify(convert(self.task2worker(vars))))

    @property
    def table_text(self):
        rank = len(self.task_shape)
        if rank == 1:
            texts = []
            for i in range(self.task_shape[0]):
                texts.append(str(self.task2worker(i)))
            return " ".join(texts)
        elif rank == 2:
            width = len(str(self.num_workers))
            fmt = f'{{:{width}d}}'
            rows = []
            for i in range(self.task_shape[0]):
                row = []
                for j in range(self.task_shape[1]):
                    row.append(fmt.format(self.task2worker([i, j])))
                rows.append(" ".join(row))
            return "\n".join(rows)
        else:
            return ""


def task_layout_compose(outer: TaskLayout, inner: TaskLayout) -> TaskLayout:
    num_workers = outer.num_workers * inner.num_workers
    assert len(outer.task_shape) == len(inner.task_shape)
    task_shape = [a * b for a, b in zip(outer.task_shape, inner.task_shape)]

    def task2worker(task_index: Sequence[Int]):
        outer_task_index = [a // inner.task_shape[i] for i, a in enumerate(task_index)]
        inner_task_index = [b % inner.task_shape[i] for i, b in enumerate(task_index)]
        return outer.task2worker(outer_task_index) * inner.num_workers + inner.task2worker(inner_task_index)

    def worker2task(worker_index: Int):
        outer_worker_index = worker_index // inner.num_workers
        inner_worker_index = worker_index % inner.num_workers
        outer_tasks = outer.worker2task(outer_worker_index)
        inner_tasks = inner.worker2task(inner_worker_index)
        tasks = []
        for outer_task in outer_tasks:
            for inner_task in inner_tasks:
                tasks.append(tuple(a * inner.task_shape[i] + b for i, (a, b) in enumerate(zip(outer_task, inner_task))))
        return tasks

    return TaskLayout(num_workers, task_shape, worker2task, task2worker)


class TaskLayoutGenerator:
    registered = []

    def get_layouts(self,
                    num_workers: Optional[int] = None,
                    task_shape: Optional[Tuple[int, ...]] = None,
                    rank: Optional[int] = None) -> Iterable[TaskLayout]:
        raise NotImplementedError()


def register_task_layout(layout: TaskLayout):
    TaskLayout.registered.append(layout)


def register_task_layout_generator(layout_generator: TaskLayoutGenerator):
    TaskLayoutGenerator.registered.append(layout_generator)


def get_task_layouts(valid_num_workers: Optional[Union[int, Sequence[int]]] = None,
                     task_shape: Optional[Sequence[int]] = None,
                     rank: Optional[int] = None) -> Iterator[TaskLayout]:
    if isinstance(valid_num_workers, int):
        valid_num_workers = [valid_num_workers]
    assert all(isinstance(v, int) for v in valid_num_workers)
    if task_shape is not None:
        assert all(isinstance(v, int) for v in task_shape)
    for idx, layout in enumerate(TaskLayout.registered):
        if valid_num_workers is not None and layout.num_workers not in valid_num_workers:
            continue
        if task_shape is not None:
            if tuple(task_shape) != tuple(layout.task_shape):
                continue
            if rank is not None and len(task_shape) != rank:
                continue
        yield layout
    for layout_generator in TaskLayoutGenerator.registered:
        for num_workers in valid_num_workers:
            layouts = layout_generator.get_layouts(num_workers, task_shape, rank)
            for layout in layouts:
                yield layout
