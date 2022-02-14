from typing import Union, Tuple, List, Type, Dict, Optional, Sequence, Iterator, Callable, Iterable, Mapping
import itertools
from hidet.utils import prod

Int = Union['Expr', int]


class TaskLayout:
    registered = []

    def __init__(self,
                 num_workers: int = None,
                 task_shape: Tuple[int, ...] = None,
                 worker2task: Optional[Callable[[Int], List[Tuple[Int, ...]]]] = None):
        self.num_workers: int = num_workers
        self.task_shape: Tuple[int, ...] = task_shape
        self.worker2task: Callable[[Int], List[Sequence[Int]]] = worker2task

    def __call__(self, w: Int) -> List[Tuple[Int, ...]]:
        return self.worker2task(w)

    def __mul__(self, other) -> 'TaskLayout':
        return ComposedTaskLayout(outer=self, inner=other)

    @staticmethod
    def row_major(task_shape: Sequence[int]):
        return GridTaskLayout(task_shape, perm=list(range(len(task_shape))))

    @staticmethod
    def column_major(task_shape: Sequence[int]):
        return GridTaskLayout(task_shape, perm=list(reversed(range(len(task_shape)))))

    @staticmethod
    def full_layout(task_shape: Sequence[int]):
        return FullTaskLayout(task_shape)

    def projection(self, dim2value: Mapping[int, Int]) -> 'TaskLayout':
        return ProjectedTaskLayout(base=self, dim2value=dim2value)


class FullTaskLayout(TaskLayout):
    def __init__(self, task_shape: Sequence[int]):
        super().__init__(num_workers=1, task_shape=task_shape, worker2task=self._worker2task)

    def _worker2task(self, w):
        ranges = [range(s) for s in self.task_shape]
        return list(itertools.product(*ranges))


class GridTaskLayout(TaskLayout):
    def __init__(self, task_shape: Sequence[int], perm: Sequence[int]):
        assert len(task_shape) == len(perm)
        super().__init__(num_workers=prod(task_shape), task_shape=task_shape, worker2task=lambda w: self._worker2task(w))
        self.perm = list(perm)
        self.bases = self._get_bases()

    def _get_bases(self):
        rank = len(self.perm)
        bases = [None] * rank
        s = 1
        for i in reversed(range(rank)):
            j = self.perm.index(i)
            bases[j] = s
            s *= self.task_shape[j]
        return bases

    def _worker2task(self, w: Int) -> List[Sequence[Int]]:
        task = []
        for mod, b in zip(self.task_shape, self.bases):
            task.append((w // b) % mod)
        return [task]


class ProjectedTaskLayout(TaskLayout):
    def __init__(self, base: TaskLayout, dim2value: Mapping[int, Int]):
        super().__init__(num_workers=base.num_workers, task_shape=base.task_shape,
                         worker2task=lambda w: self._worker2task(w))
        self.base = base
        self.dim2value = dim2value

    def _worker2task(self, w: Int) -> List[Sequence[Int]]:
        rank = len(self.task_shape)
        projected_tasks = []
        for task in self.base(w):
            projected_tasks.append(tuple(self.dim2value[i] if i in self.dim2value else task[i] for i in range(rank)))
        return projected_tasks


class ComposedTaskLayout(TaskLayout):
    def __init__(self, outer: TaskLayout, inner: TaskLayout):
        assert len(outer.task_shape) == len(inner.task_shape)
        super().__init__(
            num_workers=outer.num_workers * inner.num_workers,
            task_shape=[a * b for a, b in zip(outer.task_shape, inner.task_shape)],
            worker2task=lambda w: self._worker2task(w)
        )
        self.outer = outer
        self.inner = inner

    def _worker2task(self, worker_index: Int) -> List[Sequence[Int]]:
        outer_worker_index = worker_index // self.inner.num_workers
        inner_worker_index = worker_index % self.inner.num_workers
        outer_tasks = self.outer.worker2task(outer_worker_index)
        inner_tasks = self.inner.worker2task(inner_worker_index)
        tasks = []
        for outer_task in outer_tasks:
            for inner_task in inner_tasks:
                tasks.append(tuple(a * self.inner.task_shape[i] + b for i, (a, b) in enumerate(zip(outer_task, inner_task))))
        return tasks


class TaskLayoutExpander:
    def __init__(self):
        from hidet.ir.stmt import LetStmt, ForStmt
        self.stmts: List[Union[LetStmt, ForStmt]] = []

    def expand(self, w: Int, task_layout: TaskLayout) -> List[Sequence[Int]]:
        vtable = {
            FullTaskLayout: self.expand_full,
            GridTaskLayout: self.expand_grid,
            ComposedTaskLayout: self.expand_composed,
            ProjectedTaskLayout: self.expand_projected,
            TaskLayout: self.expand_atom,
        }
        # noinspection PyArgumentList
        return vtable[task_layout.__class__](w, task_layout)

    def expand_composed(self, w: Int, layout: ComposedTaskLayout):
        from hidet.ir import var, LetStmt
        outer_w = var('p')
        inner_w = var('q')
        self.stmts.append(LetStmt(outer_w, w // layout.inner.num_workers))
        self.stmts.append(LetStmt(inner_w, w % layout.inner.num_workers))
        outer_fields = self.expand(outer_w, layout.outer)
        inner_fields = self.expand(inner_w, layout.inner)
        fields = []
        for outer_field in outer_fields:
            for inner_field in inner_fields:
                fields.append(tuple(a * layout.inner.task_shape[i] + b for i, (a, b) in enumerate(zip(outer_field, inner_field))))
        return fields

    def expand_projected(self, w: Int, layout: ProjectedTaskLayout):
        rank = len(layout.task_shape)
        base_fields = self.expand(w, layout.base)
        projected_fields = []
        for field in base_fields:
            projected_fields.append(tuple(layout.dim2value[i] if i in layout.dim2value else field[i] for i in range(rank)))
        return projected_fields

    @staticmethod
    def expand_grid(w: Int, layout: GridTaskLayout):
        return layout(w)

    def expand_full(self, w: Int, layout: FullTaskLayout):
        unroll_limit = 1024
        if prod(layout.task_shape) < unroll_limit:
            # unroll automatically
            return layout(w)
        else:
            # do not expand, use for loop
            from hidet.ir import var, ForStmt
            shape = layout.task_shape
            axes = []
            for i, s in enumerate(shape):
                axis = var(chr(ord('i') + i))
                self.stmts.append(ForStmt(loop_var=axis, extent=s))
                axes.append(axis)
            return [axes]

    @staticmethod
    def expand_atom(w: Int, layout: TaskLayout):
        return layout(w)


def row_major_layout(*task_shape: Sequence[int]):
    return GridTaskLayout(task_shape, perm=list(range(len(task_shape))))


def col_major_layout(*task_shape: Sequence[int]):
    return GridTaskLayout(task_shape, perm=list(reversed(range(len(task_shape)))))


def full_layout(*task_shape: Sequence[int]):
    return FullTaskLayout(task_shape)
