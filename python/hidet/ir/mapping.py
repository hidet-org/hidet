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
# pylint: disable=import-outside-toplevel
from __future__ import annotations
from typing import Union, Tuple, List, Optional, Sequence, Callable, Mapping
import itertools
import numpy as np
from hidet.utils import prod, gcd

Int = Union['Expr', int]


def is_atom(expr):
    from hidet.ir import Constant, Var

    return isinstance(expr, (Constant, Var))


def var(hint):
    from hidet import ir

    return ir.var(hint)


def strides_from_ranks(shape: Sequence[int], ranks: Sequence[int]) -> List[int]:
    if any(v < 0 for v in shape):
        raise ValueError('Shape must be non-negative, got {}'.format(shape))
    if len(set(ranks)) != len(ranks):
        raise ValueError('Duplicated ranks: {}'.format(ranks))
    if any(v < 0 or v >= len(shape) for v in ranks):
        raise ValueError('Ranks {} out of bound for shape {}'.format(ranks, shape))
    if len(ranks) != len(shape):
        raise ValueError('Ranks must have the same length as shape, got shape {} and ranks {}'.format(shape, ranks))
    strides: List[Optional[int]] = [None] * len(shape)
    acc = 1
    for i in reversed(range(len(shape))):
        dim = ranks.index(i)
        strides[dim] = acc
        acc *= shape[dim]
    return strides


class TaskMapping:
    registered = []

    def __init__(
        self,
        num_workers: int = None,
        task_shape: Tuple[int, ...] = None,
        worker2task: Optional[Callable[[Int], List[Tuple[Int, ...]]]] = None,
    ):
        from hidet.ir import Expr
        from hidet.ir.functors import simplify_to_int

        if isinstance(num_workers, Expr):
            num_workers = simplify_to_int(num_workers)
        task_shape = tuple(simplify_to_int(v) for v in task_shape)
        self.num_workers: int = num_workers
        self.task_shape: Tuple[int, ...] = task_shape
        self.worker2task: Callable[[Int], List[Tuple[Int]]] = worker2task
        if num_workers is not None:
            assert isinstance(num_workers, int)
        if task_shape is not None:
            assert all(isinstance(s, int) for s in task_shape)

    def __call__(self, w: Int) -> List[Tuple[Int, ...]]:
        return self.worker2task(w)

    def __mul__(self, other) -> 'TaskMapping':
        return ComposedTaskMapping(outer=self, inner=other)

    def __getitem__(self, w: Int) -> List[Tuple[Int, ...]]:
        return self.worker2task(w)

    def __str__(self):
        worker_id = np.empty(shape=self.task_shape, dtype=np.int32)
        for w in range(self.num_workers):
            for task_indices in self.worker2task(w):
                worker_id[task_indices] = w
        return np.array2string(worker_id)

    def on(self, w: Int) -> List[Tuple[Int, ...]]:
        return self.worker2task(w)

    def map(self, w: Int) -> Tuple[Int, ...]:
        return self.single_task_of(w)

    def single_task_of(self, w: Int) -> Tuple[Int, ...]:
        tasks = self.worker2task(w)
        if len(tasks) != 1:
            raise ValueError('Expect to have a single task, but got {} tasks'.format(len(tasks)))
        return tasks[0]

    @staticmethod
    def row_major(task_shape: Sequence[int]):
        return SpatialTaskMapping(task_shape, ranks=list(range(len(task_shape))))

    @staticmethod
    def column_major(task_shape: Sequence[int]):
        return SpatialTaskMapping(task_shape, ranks=list(reversed(range(len(task_shape)))))

    @staticmethod
    def full_layout(task_shape: Sequence[int]):
        return RepeatTaskMapping(task_shape, ranks=list(range(len(task_shape))))

    def projection(self, dim2value: Mapping[int, Int]) -> 'TaskMapping':
        return ProjectedTaskMapping(base=self, dim2value=dim2value)

    # chain api
    def spatial(self, *task_shape, ranks: List[int] = None) -> TaskMapping:
        return self * spatial_map(task_shape, ranks)

    def repeat(self, *task_shape, ranks: List[int] = None) -> TaskMapping:
        return self * repeat_map(task_shape, ranks)


class RepeatTaskMapping(TaskMapping):
    def __init__(self, task_shape: Sequence[int], ranks: Optional[Sequence[int]]):
        if ranks is None:
            ranks = list(range(len(task_shape)))
        self.ranks: List[int] = list(ranks)
        self.strides: List[int] = strides_from_ranks(task_shape, ranks)
        super().__init__(num_workers=1, task_shape=tuple(task_shape), worker2task=self._worker2task)

    # noinspection PyUnusedLocal
    def _worker2task(self, w: Int) -> List[Tuple[Int]]:  # pylint: disable=unused-argument
        def key_func(task: Tuple[int]) -> int:
            global_index = sum(a * b for a, b in zip(task, self.strides))
            return global_index

        ranges = [range(s) for s in self.task_shape]
        tasks = list(tuple(task) for task in itertools.product(*ranges))
        return list(sorted(tasks, key=key_func))


class SpatialTaskMapping(TaskMapping):
    def __init__(self, task_shape: Sequence[int], ranks: Sequence[int]):
        assert len(task_shape) == len(ranks)
        super().__init__(num_workers=prod(task_shape), task_shape=tuple(task_shape), worker2task=self._worker2task)
        self.ranks = list(ranks)
        self.strides = strides_from_ranks(task_shape, ranks)

    def _worker2task(self, w: Int) -> List[Tuple[Int]]:
        task = []
        for mod, b in zip(self.task_shape, self.strides):
            task.append((w // b) % mod)
        return [tuple(task)]


class ProjectedTaskMapping(TaskMapping):
    def __init__(self, base: TaskMapping, dim2value: Mapping[int, Int]):
        assert all(int(v) == 0 for v in dim2value.values())
        task_shape = tuple(base.task_shape[i] if i not in dim2value else 1 for i in range(len(base.task_shape)))
        super().__init__(num_workers=base.num_workers, task_shape=task_shape, worker2task=self._worker2task)
        self.base = base
        self.dim2value = dim2value

    def _worker2task(self, w: Int) -> List[Tuple[Int]]:
        rank = len(self.task_shape)
        projected_tasks = []
        for task in self.base(w):
            projected_tasks.append(tuple(self.dim2value[i] if i in self.dim2value else task[i] for i in range(rank)))
        return projected_tasks


class ComposedTaskMapping(TaskMapping):
    def __init__(self, outer: TaskMapping, inner: TaskMapping):
        assert len(outer.task_shape) == len(inner.task_shape)
        super().__init__(
            num_workers=outer.num_workers * inner.num_workers,
            task_shape=tuple(a * b for a, b in zip(outer.task_shape, inner.task_shape)),
            worker2task=self._worker2task,
        )
        self.outer = outer
        self.inner = inner

    def _worker2task(self, worker_index: Int) -> List[Tuple[Int, ...]]:
        outer_worker_index = worker_index // self.inner.num_workers
        inner_worker_index = worker_index % self.inner.num_workers
        outer_tasks = self.outer.worker2task(outer_worker_index)
        inner_tasks = self.inner.worker2task(inner_worker_index)
        tasks = []
        for outer_task in outer_tasks:
            for inner_task in inner_tasks:
                task = tuple(a * self.inner.task_shape[i] + b for i, (a, b) in enumerate(zip(outer_task, inner_task)))
                tasks.append(task)
        return tasks


class TaskMappingExpander:
    def __init__(self):
        from hidet.ir.stmt import ForStmt, LetStmt

        self.stmts: List[Union[LetStmt, ForStmt]] = []

    def variablize(self, e):
        from hidet.ir import LetStmt

        if is_atom(e):
            return e
        else:
            v = var('p')
            self.stmts.append(LetStmt(v, e))
            return v

    def expand(self, w: Int, task_layout: TaskMapping) -> List[Sequence[Int]]:
        vtable = {
            RepeatTaskMapping: self.expand_full,
            SpatialTaskMapping: self.expand_grid,
            ComposedTaskMapping: self.expand_composed,
            ProjectedTaskMapping: self.expand_projected,
            TaskMapping: self.expand_atom,
        }
        w = self.variablize(w)
        # noinspection PyArgumentList
        return vtable[task_layout.__class__](w, task_layout)

    def expand_composed(self, w: Int, layout: ComposedTaskMapping):
        outer_w = self.variablize(w // layout.inner.num_workers)
        inner_w = self.variablize(w % layout.inner.num_workers)
        outer_fields = self.expand(outer_w, layout.outer)
        inner_fields = self.expand(inner_w, layout.inner)
        fields = []
        for outer_field in outer_fields:
            scaled_outer_field = [self.variablize(a * b) for a, b in zip(outer_field, layout.inner.task_shape)]
            for inner_field in inner_fields:
                fields.append(tuple(a + b for a, b in zip(scaled_outer_field, inner_field)))
        return fields

    def expand_projected(self, w: Int, layout: ProjectedTaskMapping):
        rank = len(layout.task_shape)
        base_fields = self.expand(w, layout.base)
        projected_fields = []
        for field in base_fields:
            projected_fields.append(
                tuple(layout.dim2value[i] if i in layout.dim2value else field[i] for i in range(rank))
            )
        return projected_fields

    def expand_grid(self, w: Int, layout: SpatialTaskMapping):
        return [[self.variablize(v) for v in layout(w)[0]]]

    def expand_full(self, w: Int, layout: RepeatTaskMapping):
        unroll_limit = 1024
        if prod(layout.task_shape) < unroll_limit:
            # unroll automatically
            return layout(w)
        else:
            # do not expand, use for loop
            from hidet.ir import ForStmt

            shape = layout.task_shape
            axes = []
            for i, s in enumerate(shape):
                axis = var(chr(ord('i') + i))
                self.stmts.append(ForStmt(loop_var=axis, extent=s))
                axes.append(axis)
            return [axes]

    @staticmethod
    def expand_atom(w: Int, layout: TaskMapping):
        return layout(w)


def spatial_map(task_shape: Sequence[int], ranks: Optional[Sequence[int]] = None):
    from hidet.ir.functors import simplify_to_int

    task_shape = [simplify_to_int(v) for v in task_shape]
    if ranks is None:
        ranks = list(range(len(task_shape)))
    return SpatialTaskMapping(task_shape, ranks)


def row_spatial(*task_shape: int):
    return spatial_map(task_shape)


def col_spatial(*task_shape: int):
    return spatial_map(task_shape, ranks=list(reversed(range(len(task_shape)))))


def repeat_map(task_shape: Sequence[int], ranks: Optional[Sequence[int]] = None):
    from hidet.ir.functors import simplify_to_int

    task_shape = [simplify_to_int(v) for v in task_shape]
    if ranks is None:
        ranks = list(range(len(task_shape)))
    return RepeatTaskMapping(task_shape, ranks)


def row_repeat(*task_shape: int):
    return repeat_map(task_shape)


def col_repeat(*task_shape: int):
    return repeat_map(task_shape, ranks=list(reversed(range(len(task_shape)))))


def auto_map(*task_shape: int, workers: int, ranks: Optional[Sequence[int]] = None) -> TaskMapping:
    task_shape: List[int] = list(task_shape)
    num_tasks = prod(task_shape)
    if num_tasks % workers != 0:
        raise ValueError(
            'Expect the number of tasks {} in task shape {} be a multiple of number of workers {}.'.format(
                num_tasks, task_shape, workers
            )
        )
    num_dims = len(task_shape)
    if ranks is None:
        ranks = list(range(num_dims))
    else:
        ranks = list(ranks)
    spatial_shape = [0] * num_dims
    repeat_shape = [0] * num_dims

    remain = workers
    for rank in reversed(range(num_dims)):
        dim = ranks.index(rank)
        spatial_shape[dim] = gcd(remain, task_shape[dim])
        repeat_shape[dim] = task_shape[dim] // spatial_shape[dim]
        remain //= spatial_shape[dim]
    return repeat_map(repeat_shape, ranks) * spatial_map(spatial_shape, ranks)
