# from typing import Tuple, List, Optional, Iterable, Union, Sequence, Iterator
# from functools import reduce, partial
# from itertools import product
# import operator
# from sympy.ntheory import divisors
#
# from .task_layout import TaskMapping, Int
#
#
# class TaskLayoutGenerator:
#     registered = []
#
#     def get_layouts(self,
#                     num_workers: Optional[int] = None,
#                     task_shape: Optional[Tuple[int, ...]] = None,
#                     rank: Optional[int] = None) -> Iterable[TaskMapping]:
#         raise NotImplementedError()
#
#
# def register_task_layout(layout: TaskMapping):
#     TaskMapping.registered.append(layout)
#
#
# def register_task_layout_generator(layout_generator: TaskLayoutGenerator):
#     TaskLayoutGenerator.registered.append(layout_generator)
#
#
# def get_task_layouts(valid_num_workers: Optional[Union[int, Sequence[int]]] = None,
#                      task_shape: Optional[Sequence[int]] = None,
#                      rank: Optional[int] = None) -> Iterator[TaskMapping]:
#     if isinstance(valid_num_workers, int):
#         valid_num_workers = [valid_num_workers]
#     assert all(isinstance(v, int) for v in valid_num_workers)
#     if task_shape is not None:
#         assert all(isinstance(v, int) for v in task_shape)
#     for idx, layout in enumerate(TaskMapping.registered):
#         if valid_num_workers is not None and layout.num_workers not in valid_num_workers:
#             continue
#         if task_shape is not None:
#             if tuple(task_shape) != tuple(layout.task_shape):
#                 continue
#             if rank is not None and len(task_shape) != rank:
#                 continue
#         yield layout
#     for layout_generator in TaskLayoutGenerator.registered:
#         for num_workers in valid_num_workers:
#             layouts = layout_generator.get_layouts(num_workers, task_shape, rank)
#             for layout in layouts:
#                 yield layout
#
#
# def decompose_integer(n, num_items):
#     """
#     decompose n into num_items items such that the product of these items equals to n. Order sensitive.
#     return a list of valid decomposition. E.g.,
#     decompose_integer(n=12, num_items=2) => [(1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)]
#     """
#     results = []
#     current_result = [None] * num_items
#
#     def helper(remaining, ith):
#         if ith + 1 == num_items:
#             current_result[ith] = remaining
#             results.append(tuple(current_result))
#             return
#         for v in divisors(remaining):
#             current_result[ith] = v
#             helper(remaining // v, ith + 1)
#
#     helper(n, 0)
#     return results
#
#
# class FullLayout(TaskLayoutGenerator):
#     @classmethod
#     def get_layouts(cls,
#                     num_workers: Optional[int] = None,
#                     task_shape: Optional[Tuple[int, ...]] = None,
#                     rank: Optional[int] = None) -> Iterable[TaskMapping]:
#         if num_workers is not None and num_workers != 1:
#             return
#         if task_shape is None:
#             return
#         yield TaskMapping(1, task_shape, worker2task=partial(cls.worker2task, task_shape=task_shape))
#
#     @staticmethod
#     def worker2task(worker_index: Int, task_shape: Tuple[int, ...]) -> List[Tuple[Int, ...]]:
#         ranges = [range(s) for s in task_shape]
#         result = list(product(*ranges))
#         return result
#
#     @staticmethod
#     def task2worker(task_index: Tuple[Int, ...], task_shape: Tuple[int, ...]) -> Int:
#         return 0
#
#
# class RowMajorLayout(TaskLayoutGenerator):
#     @classmethod
#     def get_layouts(cls,
#                     num_workers: Optional[int] = None,
#                     task_shape: Optional[Tuple[int, ...]] = None,
#                     rank: Optional[int] = None) -> Iterable[TaskMapping]:
#         if num_workers is None and task_shape is None:
#             return
#         elif num_workers is None:
#             num_workers = reduce(operator.mul, task_shape)
#             task_shapes = [task_shape]
#         elif task_shape is None:
#             assert rank is not None
#             task_shapes = decompose_integer(num_workers, rank)
#         else:
#             assert num_workers == reduce(operator.mul, task_shape)
#             task_shapes = [task_shape]
#
#         for task_shape in task_shapes:
#             yield TaskMapping(num_workers, task_shape, partial(cls.worker2task, task_shape=task_shape))
#
#     @staticmethod
#     def worker2task(worker_index: Int, task_shape: Tuple[Int, ...]) -> List[Tuple[Int, ...]]:
#         task_index = []
#         rank = len(task_shape)
#         bases = [reduce(operator.mul, task_shape[i + 1:], 1) for i in range(rank)]
#         for i in range(rank):
#             task_index.append(worker_index // bases[i] % task_shape[i])
#         return [tuple(task_index)]
#
#     @staticmethod
#     def task2worker(task_index: Tuple[Int, ...], task_shape: Tuple[Int, ...]) -> Int:
#         worker_index = 0
#         rank = len(task_shape)
#         bases = [reduce(operator.mul, task_shape[i + 1:], 1) for i in range(rank)]
#         for i in range(rank):
#             worker_index += task_index[i] * bases[i]
#         return worker_index
#
#
# class ColumnMajorLayout(TaskLayoutGenerator):
#     @classmethod
#     def get_layouts(cls,
#                     num_workers: Optional[int] = None,
#                     task_shape: Optional[Tuple[int, ...]] = None,
#                     rank: Optional[int] = None) -> Iterable[TaskMapping]:
#         if num_workers is None and task_shape is None:
#             return
#         elif num_workers is None:
#             num_workers = reduce(operator.mul, task_shape)
#             task_shapes = [task_shape]
#         elif task_shape is None:
#             task_shapes = decompose_integer(num_workers, rank)
#         else:
#             assert num_workers == reduce(operator.mul, task_shape)
#             task_shapes = [task_shape]
#
#         for task_shape in task_shapes:
#             yield TaskMapping(num_workers, task_shape,
#                               partial(cls.worker2task, task_shape=task_shape))
#
#     @staticmethod
#     def worker2task(worker_index: Int, task_shape) -> List[Tuple[Int, ...]]:
#         task_index = []
#         rank = len(task_shape)
#         bases = [reduce(operator.mul, task_shape[:i], 1) for i in range(rank)]
#         for i in range(rank):
#             task_index.append(worker_index // bases[i] % task_shape[i])
#         return [tuple(task_index)]
#
#     @staticmethod
#     def task2worker(task_index: Tuple[Int, ...], task_shape) -> Int:
#         worker_index = 0
#         rank = len(task_shape)
#         bases = [reduce(operator.mul, task_shape[:i], 1) for i in range(rank)]
#         for i in range(rank):
#             worker_index += task_index[i] * bases[i]
#         return worker_index
#
#
# def full_layout(*task_shape) -> TaskMapping:
#     layouts = list(FullLayout.get_layouts(task_shape=task_shape))
#     assert len(layouts) == 1
#     return layouts[0]
#
#
# def row_major_layout(*task_shape) -> TaskMapping:
#     layouts = list(RowMajorLayout.get_layouts(task_shape=task_shape))
#     assert len(layouts) == 1
#     return layouts[0]
#
#
# def col_major_layout(*task_shape) -> TaskMapping:
#     layouts = list(ColumnMajorLayout.get_layouts(task_shape=task_shape))
#     assert len(layouts) == 1
#     return layouts[0]
