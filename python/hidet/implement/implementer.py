from collections import defaultdict
from typing import Type, Dict, Mapping
from hidet.ir.task import Task
from hidet.ir.func import IRModule
from hidet.ir.node import Node
from hidet.ir.dialects.pattern import TaskPattern, match


class Implementer:
    def priority(self) -> int:
        raise NotImplementedError()

    def task_pattern(self) -> TaskPattern:
        raise NotImplementedError()

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        raise NotImplementedError()


_implementers: Dict[int, Dict[str, Implementer]] = defaultdict(dict)


def register_impl(name):
    if name in _implementers:
        raise KeyError("Implementer has existed.")

    def wrapper(cls: Type[Implementer]):
        impl = cls()
        _implementers[impl.priority()][name] = impl
        return cls

    return wrapper


def implement(task: Task) -> IRModule:
    priorities = sorted(_implementers.keys())
    for p in reversed(priorities):
        for impl in _implementers[p].values():
            matching = match(impl.task_pattern(), task)
            if matching:
                return impl.implement(task, matching)
    raise NotImplementedError("Can not find matching implementer.")

