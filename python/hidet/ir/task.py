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
from typing import Any, Dict, List, Union, Callable, Optional, Tuple
import os
import pickle
from hidet.ir.node import Node
from hidet.ir.type import FuncType, VoidType
from hidet.ir.expr import Expr, Var, SymbolVar, var, is_constant
from hidet.ir.module import IRModule
from hidet.ir.compute import ComputeNode, TensorNode, TensorInput, ScalarInput, GridCompute
from hidet.ir.target import Target


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

    def __init__(self, name, inputs, outputs, *, inverse_map=None, attributes=None):
        inverse_map = inverse_map if inverse_map else {}
        attributes = attributes if attributes else {}
        self.name: str = name
        self.inputs: List[TensorInput] = list(inputs)
        self.outputs: List[TensorNode] = list(outputs)
        self.inverse_map: Dict[TensorInput, InverseMap] = {a: InverseMap.from_obj(b) for a, b in inverse_map.items()}
        self.attrs: Dict[str, Union[str, float, int, bool]] = attributes
        self.assertions: List[Tuple[Expr, Optional[str]]] = getattr(self, 'assertions', [])

        from hidet.ir.tools import collect

        self.symbols: List[SymbolVar] = list(collect(self.outputs, SymbolVar))
        self._sanity_check()

    def _assert(self, expr: Union[Expr, bool], msg: Optional[str] = None):
        import hidet

        simplified = hidet.ir.tools.simplify(expr)
        if is_constant(simplified):
            assert simplified, msg
        else:
            if hasattr(self, 'assertions'):
                self.assertions.append((expr, msg))
            else:
                self.assertions = [(expr, msg)]

    @property
    def params(self) -> List[TensorNode]:
        return [*self.inputs, *self.outputs]

    def _sanity_check(self):
        from hidet.ir.tools import collect_free_vars, collect

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

        free_vars: List[Var] = collect_free_vars(self.outputs)
        if any(
            v not in self.params and not isinstance(v.type, FuncType) and not isinstance(v, SymbolVar)
            for v in free_vars
        ):
            raise ValueError('Some free variables are not in params: {}'.format(free_vars))

        # check all TensorInput used in outputs are placed in inputs
        used_inputs = collect(self.outputs, TensorInput)
        if any(x not in self.inputs for x in used_inputs):
            raise ValueError('Some TensorInput used in outputs are not placed in inputs: {}'.format(used_inputs))

        # check assertions for correctness
        assert_symbols: List[SymbolVar] = list(collect([cond for cond, _ in self.assertions], SymbolVar))
        for sym in assert_symbols:
            assert sym in self.symbols, f"encountered {sym} in assertions, but not in list of defined symbols"

    def has_symbolic_shape(self) -> bool:
        from hidet.ir.tools import collect

        return len(collect(self.outputs, SymbolVar)) > 0

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
                    arguments.append(hidet.zeros(param.const_shape, dtype=param.type.dtype, device=device))
                elif param.type.dtype.is_float():
                    arguments.append(hidet.randn(param.const_shape, dtype=param.type.dtype, device=device))
                else:
                    raise ValueError('Unknown dtype: {}'.format(param.type.dtype))
            else:
                raise ValueError('Unknown parameter type: {}'.format(type(param)))
        return arguments

    def build(self, target: Union[str, Target], load: bool = True):
        """
        Build the task for the given target to a callable function.

        Parameters
        ----------
        target: Union[str, Target]
            The target device.

        load: bool
            Whether to load the task
        Returns
        -------
        func: hidet.runtime.CompiledTask
            The compiled module.
        """
        from hidet.drivers import build_task

        if isinstance(target, Target):
            target = target.name
        return build_task(self, target=target, load=load)

    def implement(self, target: Union[Target, str], working_dir: str) -> List[IRModule]:
        from hidet.ir.schedulers import CudaAutoScheduler, CpuAutoScheduler

        if isinstance(target, str):
            target = Target.from_string(target)

        implement_target, scheduler = {
            'cuda': (self.implement_cuda, CudaAutoScheduler),
            'cpu': (self.implement_cpu, CpuAutoScheduler),
        }[target.name]
        ir_modules: Union[IRModule, List[IRModule]] = implement_target(working_dir)
        if ir_modules is NotImplemented:
            auto_scheduler = scheduler()
            ir_modules = [auto_scheduler.schedule_task(self, target.name)]
        elif isinstance(ir_modules, IRModule):
            ir_modules = [ir_modules]
        elif isinstance(ir_modules, (list, tuple)) and all(isinstance(x, IRModule) for x in ir_modules):
            ir_modules = list(ir_modules)
        else:
            raise ValueError(
                'Expect the `implement` method to return an IRModule or List[IRModule], got {}'.format(ir_modules)
            )

        return ir_modules

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


def save_task(task: Task, fname: str):
    task.save(fname)


def load_task(fname: str) -> Task:
    return Task.load(fname)


def task_compiled_func_type(task: Task) -> FuncType:
    from hidet.ir.tools import infer_type

    return FuncType(param_types=[infer_type(t) for t in task.params], ret_type=VoidType())
