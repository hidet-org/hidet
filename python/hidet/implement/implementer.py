from typing import Optional
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


def implement(task: Task, impl_name: Optional[str] = None) -> IRModule:
    priorities = sorted(_implementers.keys())
    messages = {}
    for p in reversed(priorities):
        for name, impl in _implementers[p].items():
            if impl_name and impl_name != name:
                continue
            matching, msg = match(impl.task_pattern(), task)
            messages[name] = msg
            if matching:
                return impl.implement(task, matching)
    report = "\n".join([f'{name}:\n{msg}' for name, msg in messages.items()])
    raise NotImplementedError(f"Can not find matching implementer for task {task.name}. Logs: \n\n {report}")

