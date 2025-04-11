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

from typing import Dict, Type, Callable, Optional, Set, List, Union
import operator
import itertools
import inspect
import torch

from hidet.graph.tensor import Tensor as HidetTensor
from .utils import tensor_from_torch


class OverloadedFunction:
    def __init__(self):
        from inspect import Signature

        self.signatures: List[Signature] = []
        self.functions: List[Callable] = []

    def __call__(self, *args, **kwargs):
        dispatched = self.resolve(*args, **kwargs)
        if dispatched is None:
            raise RuntimeError('Can not dispatch function')
        return dispatched(*args, **kwargs)

    @staticmethod
    def from_lambda(func: Callable):
        of = OverloadedFunction()
        of.overload(func)
        return of

    def resolve(self, *args, **kwargs) -> Optional[Callable]:
        for sig, func in zip(self.signatures, self.functions):
            try:
                sig.bind(*args, **kwargs)
            except TypeError:
                continue
            else:
                return func
        return None

    def overload(self, func: Callable):
        self.functions.append(func)
        self.signatures.append(inspect.signature(func))
        return self


class Registry:
    # registered modules, like torch.nn.Conv2d, torch.nn.Linear.
    registered_modules: Dict[Type[torch.nn.Module], Type['HidetModule']] = {}
    # registered functions, like torch.add, torch.mul, torch.nn.functional.relu, and torch.ops.aten.cos.
    registered_functions: Dict[Callable, OverloadedFunction] = {}
    # registered methods, like torch.Tensor.add, torch.Tensor.mul, torch.Tensor.relu.
    registered_methods: Dict[Callable, OverloadedFunction] = {}


class ExpectedRegistry:
    operator_functions: Set[Callable] = set(c for c in operator.__dict__.values() if callable(c))
    torch_functional: Set[Callable] = set(c for c in torch.nn.functional.__dict__.values() if callable(c))
    torch_modules: Set[Type[torch.nn.Module]] = set(
        c for c in torch.nn.__dict__.values() if isinstance(c, type) and issubclass(c, torch.nn.Module)
    )
    torch_root_functions: Set[Callable] = set(c for c in torch.__dict__.values() if callable(c))
    torch_tensor_methods: Set[Callable] = set(
        getattr(torch.Tensor, name) for name in dir(torch.Tensor) if callable(getattr(torch.Tensor, name))
    )


def register_module(torch_cls: Type[torch.nn.Module]):
    def decorator(hidet_cls: Type[HidetModule]):
        Registry.registered_modules[torch_cls] = hidet_cls
        return hidet_cls

    return decorator


def register_function(func: Union[Callable, str]):
    def decorator(hidet_func):
        if isinstance(func, str):
            try:
                nfunc = eval(func)  # pylint: disable=eval-used
            except AttributeError:
                # No function with such name
                return hidet_func
        else:
            nfunc = func
        if nfunc not in Registry.registered_functions:
            Registry.registered_functions[nfunc] = OverloadedFunction()
        Registry.registered_functions[nfunc].overload(hidet_func)
        return hidet_func

    return decorator


def register_method(method: Callable):
    def decorator(hidet_method):
        if method not in Registry.registered_methods:
            Registry.registered_methods[method] = OverloadedFunction()
        Registry.registered_methods[method].overload(hidet_method)
        return hidet_method

    return decorator


class HidetModule:
    def __init__(self, torch_module: torch.nn.Module):
        self.mod: torch.nn.Module = torch_module
        self.torch_params: Dict[str, Optional[torch.Tensor]] = dict(
            itertools.chain(self.mod.named_parameters(), self.mod.named_buffers())
        )
        self.hidet_params: Dict[str, Optional[HidetTensor]] = {}

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def _get_weight_norm_hook(self, name: str):
        from torch.nn.utils.weight_norm import WeightNorm

        for hook in self.mod._forward_pre_hooks.values():  # pylint: disable=protected-access
            if isinstance(hook, WeightNorm) and hook.name == name:
                return hook
        return None

    def _used_weight_norm(self, name: str) -> bool:
        return self._get_weight_norm_hook(name) is not None

    def _compute_weight_norm(self, name: str) -> HidetTensor:
        hook = self._get_weight_norm_hook(name)
        return hook.compute_weight(self.mod)

    def param(self, name: str, optional=False, steal=False) -> Optional[HidetTensor]:
        if name not in self.torch_params:
            # see https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
            # to learn more about weight norm.
            if self._used_weight_norm(name):
                self.torch_params[name] = self._compute_weight_norm(name)
                return self.param(name, optional)

            if optional:
                return None
            raise RuntimeError(f"hidet: {self.mod} has no parameter/buffer {name}")

        if name not in self.hidet_params:
            if self.torch_params[name] is None:
                self.hidet_params[name] = None
            else:
                torch_param: torch.Tensor = self.torch_params[name]
                if steal:
                    del self.torch_params[name]
                    setattr(self.mod, name, None)

                # Force the memory of the parameter to be managed by hidet.
                # The graph optimization may create new parameters during the
                # graph transformation. If the storage of the parameter is managed
                # by other libraries, the memory won't be freed after the graph
                # optimization, which will cause OOM error.
                def tensor_clone_from_torch(tensor: torch.Tensor):
                    hidet_tensor = tensor_from_torch(tensor)
                    from hidet.graph import empty_like

                    hidet_param = empty_like(hidet_tensor)
                    hidet_param.copy_(hidet_tensor)
                    del hidet_tensor
                    return hidet_param

                if steal:
                    self.hidet_params[name] = tensor_clone_from_torch(torch_param.contiguous())
                else:
                    self.hidet_params[name] = tensor_from_torch(torch_param.contiguous())
                del torch_param
                torch.cuda.empty_cache()
        return self.hidet_params[name]


TRACED_MODULES = {'torch'}


# Allow in fxgraph registered functions only.
def allow_in_graph_registered_funcs_only():
    import sys

    # Have to import these to register everthing
    from . import register_functions, register_modules, register_methods  # pylint: disable=unused-import

    from packaging import version

    # torch._dynamo.trace_rules was introduced in v2.2.0
    if version.parse(torch.__version__) < version.parse("2.2.0"):
        return

    from torch._dynamo import variables, disallow_in_graph
    from torch._dynamo.trace_rules import lookup, lookup_callable, _allowed_callable_ids

    def warmup_dissallow():
        for module_name, module in sys.modules.items():
            if module_name.split('.')[0] not in TRACED_MODULES:
                continue
            for obj in vars(module).values():
                if not callable(obj):
                    continue
                # This check is required to following `lookup()` call.
                # It get the __name__ attribute without checking if it exists.
                if not hasattr(obj, '__name__'):
                    continue
                # This check is duplication from `torch._dynamo.disallow_in_graph`
                # We can(and should) process callables those can be disallowed in graph
                if (
                    lookup_callable(obj) != variables.TorchInGraphFunctionVariable
                    and lookup(obj) != variables.TorchInGraphFunctionVariable
                ):
                    continue
                if obj in Registry.registered_functions:
                    continue
                disallow_in_graph(obj)
                return

    warmup_dissallow()

    for module_name, module in sys.modules.items():
        if module_name.split('.')[0] not in TRACED_MODULES:
            continue
        for obj in vars(module).values():
            if not callable(obj):
                continue
            # This check is required to following `lookup()` call.
            # It get the __name__ attribute without checking if it exists.
            if not hasattr(obj, '__name__'):
                continue
            # This check is duplication from `torch._dynamo.disallow_in_graph`
            # We can(and should) process callables those can be disallowed in graph
            if (
                lookup_callable(obj) != variables.TorchInGraphFunctionVariable
                and lookup(obj) != variables.TorchInGraphFunctionVariable
            ):
                continue
            if obj in Registry.registered_functions:
                continue
            disallow_in_graph(obj)
            # print(f"{obj.__module__}.{obj.__qualname__}")

    new_func_ids = set()
    for registered_func in Registry.registered_functions:
        our_id = id(registered_func)
        if our_id in _allowed_callable_ids.function_ids:  # pylint: disable=protected-access
            new_func_ids.add(our_id)
    _allowed_callable_ids.function_ids = new_func_ids  # pylint: disable=protected-access
