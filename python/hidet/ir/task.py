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
from typing import Any, Dict, List, Union, Callable
import os
import pickle
from hidet.ir.node import Node
from hidet.ir.type import FuncType, VoidType
from hidet.ir.expr import Expr, Var, var
from hidet.ir.func import IRModule
from hidet.ir.compute import ComputeNode, TensorNode, TensorInput, ScalarInput, GridCompute


class Target:
    _supported_targets = ['cuda', 'cpu']

    def __init__(self, name: str, attrs: List[str]):
        if name not in self._supported_targets:
            raise ValueError('Does not support target {}, candidates {}.'.format(name, self._supported_targets))
        self.name = name
        self.attrs = attrs

    @staticmethod
    def from_string(target_string: str) -> Target:
        items = target_string.split()
        name, attrs = items[0], items[1:]
        return Target(name, attrs)


class InverseMap(Node):
    def __init__(self, axes: List[Var], indices: List[Expr]):
        from hidet.ir.tools import simplify

        self.axes: List[Var] = axes
        self.indices: List[Expr] = [simplify(e) for e in indices]

    @staticmethod
    def from_obj(obj: Union[InverseMap, Callable[[Any], Any]]):
        if isinstance(obj, InverseMap):
            return obj
        else:
            return InverseMap.from_lambda(lambda *args: obj(*args))  # pylint: disable=unnecessary-lambda

    @staticmethod
    def from_lambda(func, num_args=None) -> InverseMap:
        from hidet.ir.utils import as_expr

        num_args = num_args if num_args is not None else func.__code__.co_argcount
        axes = [var('v') for v in range(num_args)]
        indices = [as_expr(index_expr) for index_expr in func(*axes)]
        return InverseMap(axes, indices)

    @staticmethod
    def identity(num_args: int) -> InverseMap:
        return InverseMap.from_lambda(lambda *indices: list(indices), num_args=num_args)

    def __add__(self, other) -> InverseMap:
        from hidet.ir.tools import rewrite

        if not isinstance(other, InverseMap):
            raise ValueError('Can not concat InverseMap with {}'.format(type(other)))
        lhs, rhs = self, other
        if len(lhs.indices) != len(rhs.axes):
            raise ValueError(
                'Can not concat InverseMap a and b, '
                'where a has {} indices and b has {} axes'.format(len(lhs.indices), len(rhs.axes))
            )
        rmap = dict(zip(rhs.axes, lhs.indices))
        indices = [rewrite(index_expr, rmap) for index_expr in rhs.indices]
        return InverseMap(lhs.axes, indices)


class Task(Node):
    """
    A task defines the operator computation.
    """

    def __init__(self, name, inputs, outputs, *, params=None, inverse_map=None, attributes=None):
        params = params if params else list(inputs) + list(outputs)
        inverse_map = inverse_map if inverse_map else {}
        attributes = attributes if attributes else {}
        self.name: str = name
        self.inputs: List[TensorInput] = list(inputs)
        self.outputs: List[TensorNode] = list(outputs)
        self.params: List[Union[Var, TensorNode]] = params
        self.inverse_map: Dict[TensorInput, InverseMap] = {a: InverseMap.from_obj(b) for a, b in inverse_map.items()}
        self.attrs: Dict[str, Union[str, float, int, bool]] = attributes
        self.specialization: Dict[Var, int] = {}

        # sanity check
        for tn, im in self.inverse_map.items():
            if len(im.axes) != tn.ndim:
                raise ValueError(
                    'InverseMap for tensor {} has {} input axes, but input tensor has {} axes'.format(
                        tn.name, len(im.axes), tn.ndim
                    )
                )
            if len(im.indices) != self.outputs[0].ndim:
                raise ValueError(
                    'InverseMap for tensor {} has {} output indices, but output tensor has {} axes'.format(
                        tn.name, len(im.indices), self.outputs[0].ndim
                    )
                )

        from hidet.ir.tools import collect_free_vars

        free_vars: List[Var] = collect_free_vars(self.outputs)
        if any(v not in self.params and not isinstance(v.type, FuncType) for v in free_vars):
            raise ValueError('Some free variables are not in params: {}'.format(free_vars))

    def signature(self) -> str:
        params = []
        for tensor in self.tensor_params:
            name = tensor.name
            dtype = tensor.type.dtype.name
            params.append('{}={}{}'.format(name, dtype, tensor.type.shape))
        for name, value in self.attrs.items():
            params.append('{}={}'.format(name, repr(value)))
        param_doc = ', '.join(params)
        fuse_doc = ''
        return ''.join([self.name, '(', param_doc, ')', fuse_doc])

    def specialize_for(self, inputs):
        """
        Specialize this task for the given inputs.

        Parameters
        ----------
        inputs: Sequence[hidet.graph.Tensor]
            The input tensors.

        Returns
        -------
        task: hidet.ir.Task
            Self task specialized for the given inputs.
        """
        remap: Dict[Var, int] = {}
        for a, b in zip(self.inputs, inputs):
            for d1, d2 in zip(a.shape, b.shape):
                if isinstance(d1, Var) and isinstance(d2, int):
                    remap[d1] = d2
        self.specialization.clear()
        for param in self.params:
            if isinstance(param, Var) and param in remap:
                self.specialization[param] = remap[param]
        return self

    def generate_arguments(self, inputs, outputs):
        """
        Generate arguments for the compiled function of this task given the tensor parameters.

        Parameters
        ----------
        inputs: Sequence[Tensor]
            The input tensors.

        outputs: Sequence[Tensor]
            The output tensors.

        Returns
        -------
        args: Sequence[Tensor or int]
            The arguments for the compiled function.
        """
        remap = {a: b for a, b in zip(self.inputs, inputs)}
        remap.update({a: b for a, b in zip(self.outputs, outputs)})
        return [remap[arg] for arg in self.params]

    @property
    def tensor_params(self) -> List[TensorNode]:
        ret: List[TensorNode] = []
        ret.extend(self.inputs)
        ret.extend(self.outputs)
        return ret

    def dummy_arguments(self, device: str):
        """
        Generate dummy arguments for the compiled function of this task.

        Parameters
        ----------
        device: str
            The target device.

        Returns
        -------
        args: Sequence[Tensor or int]
            The arguments for the compiled function.
        """
        import hidet
        from hidet.graph.tensor import Tensor

        arguments: List[Union[Tensor, int]] = []
        for param in self.params:
            if isinstance(param, Var):
                arguments.append(10)
            elif isinstance(param, TensorNode):
                if param.type.dtype.is_integer():
                    arguments.append(hidet.zeros(param.const_shape(), dtype=param.type.dtype, device=device))
                elif param.type.dtype.is_float():
                    arguments.append(hidet.randn(param.const_shape(), dtype=param.type.dtype, device=device))
                else:
                    raise ValueError('Unknown dtype: {}'.format(param.type.dtype))
            else:
                raise ValueError('Unknown parameter type: {}'.format(type(param)))
        return arguments

    def build(self, target: Union[str, Target]):
        """
        Build the task for the given target to a callable function.

        Parameters
        ----------
        target: Union[str, Target]
            The target device.

        Returns
        -------
        func: hidet.runtime.CompiledFunction
            The compiled function.
        """
        from hidet.driver import build_task

        if isinstance(target, Target):
            target = target.name
        return build_task(self, target_device=target, load=True)

    def implement(self, target: Union[Target, str], working_dir: str) -> IRModule:
        from hidet.graph.ops.schedules.cuda.auto_scheduler import CudaAutoScheduler
        from hidet.graph.ops.schedules.cpu.auto_scheduler import CpuAutoScheduler
        from hidet.graph.ops.definitions.utils.tune import tune

        if isinstance(target, str):
            target = Target.from_string(target)
        if target.name == 'cuda':
            ret = self.implement_cuda(working_dir)
            if ret is NotImplemented:
                auto_scheduler = CudaAutoScheduler()
                ret = auto_scheduler.schedule_task(self, 'cuda')
        elif target.name == 'cpu':
            ret = self.implement_cpu(working_dir)
            if ret is NotImplemented:
                auto_scheduler = CpuAutoScheduler()
                ret = auto_scheduler.schedule_task(self, 'cpu')
        else:
            raise ValueError()
        if isinstance(ret, IRModule):
            return ret

        ir_modules: List[IRModule] = ret

        if len(ir_modules) == 1:
            return ir_modules[0]

        if not all(isinstance(m, IRModule) for m in ir_modules):
            raise AssertionError(
                f'The task implement function should return an IRModule or a sequence of IRModule, '
                f'but got a {type(ir_modules)}.'
            )
        dummy_args = self.dummy_arguments(target.name)
        try:
            best_ir_module = tune(ir_modules, dummy_inputs=dummy_args, working_dir=working_dir)
        except ValueError as e:
            raise RuntimeError(f'Failed to tune the task: {self}') from e
        return best_ir_module

    def implement_cuda(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return NotImplemented

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return NotImplemented

    def allow_prologue(self) -> bool:
        return True

    def allow_epilogue(self) -> bool:
        return True

    def is_injective(self) -> bool:
        from hidet.ir.tools import collect

        allowed_nodes = (ScalarInput, TensorInput, GridCompute)
        # if found other node like ReduceCompute and ArgReduceCompute, return False
        found_nodes = collect(self.outputs, ComputeNode, stop_when_found=False)
        return all(isinstance(node, allowed_nodes) for node in found_nodes)

    def is_bijective(self) -> bool:
        return self.is_injective() and len(self.inverse_map) > 0

    def save(self, fname: str):
        dirname = os.path.dirname(fname)
        os.makedirs(dirname, exist_ok=True)
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname: str) -> Task:
        with open(fname, 'rb') as f:
            return pickle.load(f)


# def is_injective_task(task: Task) -> bool:
#     """
#     Check whether a task is an injective task. A task is injective if and only if there is no reduce compute in
#     the task.
#
#     Parameters
#     ----------
#     task: Task
#         The task to check.
#
#     Returns
#     -------
#     ret: bool
#         Whether the task is injective.
#     """
#     from hidet.ir.tools import collect
#
#     scalar_nodes: List[ScalarNode] = collect(task.outputs, ScalarNode, stop_when_found=False)
#     return all(sn.scalar_compute is None for sn in scalar_nodes)
#
#
# def is_unary_injective_task(task: Task) -> bool:
#     return len(task.inputs) == 1 and len(task.outputs) == 1 and is_injective_task(task)
#
#
# def is_elementwise_task(task: Task) -> bool:
#     """
#     Check whether a task is an elementwise task. A task is elementwise if and only if it is a unary injective and
#     invertible.
#
#     Parameters
#     ----------
#     task: Task
#         The task to check.
#
#     Returns
#     -------
#     ret: bool
#         Whether the task is elementwise.
#     """
#     return is_unary_injective_task(task) and len(task.inverse_map) > 0


def save_task(task: Task, fname: str):
    task.save(fname)


def load_task(fname: str) -> Task:
    return Task.load(fname)


def task_compiled_func_type(task: Task) -> FuncType:
    from hidet.ir.tools import infer_type

    return FuncType(param_types=[infer_type(t) for t in task.params], ret_type=VoidType())
