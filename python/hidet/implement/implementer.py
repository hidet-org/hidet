from typing import Optional
from textwrap import indent
from collections import defaultdict
from typing import Type, Dict, Mapping
from hidet.ir.task import Task
from hidet.ir.func import IRModule
from hidet.ir.node import Node
from hidet.ir.dialects.pattern import TaskPattern, match


class NotSupportedError(Exception):
    pass


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
    match_messages = {}
    impl_messages = {}
    ir_module = IRModule(funcs={}, task=task)
    for p in reversed(priorities):
        for name, impl in _implementers[p].items():
            if impl_name and impl_name != name:
                continue
            matching, msg = match(impl.task_pattern(), task)
            if matching:
                try:
                    impl_ir_module = impl.implement(task, matching)
                except NotSupportedError as e:
                    impl_messages[name] = str(e)
                else:
                    if len(impl_ir_module.functions) == 0:
                        impl_messages[name] = 'Empty ir module returned.'
                    else:
                        ir_module.include(impl_ir_module)
            else:
                impl_worker_cls = impl.task_pattern().worker.__class__
                task_worker_cls = task.worker.__class__
                if impl_worker_cls == task_worker_cls:
                    match_messages[name] = msg
        if len(ir_module.functions) != 0:
            return ir_module
    match_report = "\n".join([f'{name}:\n{indent(msg, "    ")}' for name, msg in match_messages.items()])
    impl_report = "\n".join([f'{name}:\n{indent(msg, "    ")}' for name, msg in impl_messages.items()])
    lines = ['Can not implement task \'{}\''.format(task.name)]
    if len(match_messages) > 0:
        lines.append('Implementers failed to match: {}'.format(str(list(match_messages.keys()))))
    if len(impl_messages) > 0:
        lines.append('Implementers matched but failed to implement: {}'.format(str(list(impl_messages.keys()))))
    if len(match_messages) > 0:
        lines.append('Detail logs for each implementer failed to match: \n{}'.format(match_report))
    if len(impl_messages) > 0:
        lines.append('Detail logs for each implementer matched but failed to implement: \n{}'.format(impl_report))

    raise NotImplementedError('\n'.join(lines))

