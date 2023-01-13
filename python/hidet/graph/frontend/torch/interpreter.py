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
from __future__ import annotations

from typing import Dict, Any, Type, Callable, Optional, Tuple, Set, List
import logging
import inspect
import operator
import itertools
import tabulate
import torch

from hidet.ir.type import data_type
from hidet.graph.tensor import Tensor
from .utils import relative_absolute_error

logger = logging.getLogger(__name__)


class Registry:
    registered_modules: Dict[Type[torch.nn.Module], Type[HidetModule]] = {}
    registered_functions: Dict[Callable, Callable] = {}
    registered_methods: Dict[Callable, Callable] = {}


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


class UniqueWarnings:
    """
    Used to suppress duplicate warnings.
    """

    def __init__(self):
        self.warned: Set[str] = set()

    def warn_once(self, msg: str):
        """
        Only warn once for all duplicated messages between resets.
        """
        import warnings as _warnings

        if msg not in self.warned:
            _warnings.warn(msg, stacklevel=2)
            self.warned.add(msg)

    @staticmethod
    def warn(msg: str):
        import warnings as _warnings

        _warnings.warn(msg, stacklevel=2)

    def reset(self):
        self.warned.clear()


warnings = UniqueWarnings()


def register_module(torch_cls: Type[torch.nn.Module]):
    def decorator(hidet_cls: Type[HidetModule]):
        Registry.registered_modules[torch_cls] = hidet_cls
        return hidet_cls

    return decorator


def register_function(func: Callable):
    def decorator(hidet_func):
        Registry.registered_functions[func] = hidet_func
        return hidet_func

    return decorator


def register_method(method: Callable):
    def decorator(hidet_method):
        Registry.registered_methods[method] = hidet_method
        return hidet_method

    return decorator


def tensor_from_torch(tensor: torch.Tensor) -> Tensor:
    import hidet.graph.tensor

    if tensor.requires_grad:
        if torch.is_grad_enabled():
            warnings.warn_once(
                "hidet: a parameter with requires_grad=True used for computation while "
                "torch.is_grad_enabled() is True.  Please consider use 'with torch.no_grad()' "
                "to wrap the model execution."
            )
        tensor = tensor.detach()
    return hidet.graph.tensor.from_torch(tensor)


class HidetModule:
    def __init__(self, torch_module: torch.nn.Module):
        self.mod: torch.nn.Module = torch_module
        self.torch_params: Dict[str, Optional[torch.Tensor]] = dict(
            itertools.chain(self.mod.named_parameters(), self.mod.named_buffers())
        )
        self.hidet_params: Dict[str, Optional[Tensor]] = {}

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def param(self, name: str, optional=False) -> Optional[Tensor]:
        if name not in self.torch_params:
            if optional:
                return None
            raise RuntimeError(f"hidet: {self.mod} has no parameter/buffer {name}")
        if name not in self.hidet_params:
            if self.torch_params[name] is None:
                self.hidet_params[name] = None
            else:
                torch_param: torch.Tensor = self.torch_params[name]
                self.hidet_params[name] = tensor_from_torch(torch_param)
        return self.hidet_params[name]


class Interpreter:
    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__()
        self.graph_module: torch.fx.GraphModule = graph_module
        self.graph: torch.fx.Graph = graph_module.graph
        self.torch_modules: Dict[str, torch.nn.Module] = dict(graph_module.named_modules())
        self.hidet_modules: Dict[str, HidetModule] = {}

        self._check_support()

    def __call__(self, *args):
        return self.forward(*args)

    @staticmethod
    def _get_callable_name(target: Callable) -> str:
        if target in ExpectedRegistry.torch_modules:
            return f'torch.nn.{target.__name__}'
        elif target in ExpectedRegistry.torch_functional:
            return f'torch.nn.functional.{target.__name__}'
        elif target in ExpectedRegistry.operator_functions:
            return f'operator.{target.__name__}'
        elif target in ExpectedRegistry.torch_root_functions:
            return f'torch.{target.__name__}'
        elif target in ExpectedRegistry.torch_tensor_methods:
            return f'torch.Tensor.{target.__name__}'
        else:
            return str(target)

    def _check_support(self):
        not_supported = set()
        for node in self.graph.nodes:
            if node.op == "call_module":
                torch_cls = type(self.torch_modules[node.target])
                if torch_cls not in Registry.registered_modules:
                    not_supported.add(torch_cls)
            elif node.op == "call_function":
                if node.target not in Registry.registered_functions:
                    not_supported.add(node.target)
        if len(not_supported) > 0:
            lines = []
            lines.append("The following modules/functions are not supported by hidet yet:")
            for target in not_supported:
                lines.append(f"  {self._get_callable_name(target)}")
            raise NotImplementedError("\n".join(lines))

    def _lookup_hidet_module(self, target: str) -> HidetModule:
        if target not in self.hidet_modules:
            torch_module = self.torch_modules[target]
            torch_cls = type(torch_module)
            hidet_cls = Registry.registered_modules[torch_cls]
            self.hidet_modules[target] = hidet_cls(torch_module)
        return self.hidet_modules[target]

    def _lookup_hidet_method(self, torch_method):
        if torch_method not in Registry.registered_methods:
            method_name = self._get_callable_name(torch_method)
            raise NotImplementedError(f"hidet: method {method_name} is not supported yet.")
        return Registry.registered_methods[torch_method]

    @staticmethod
    def _raise_exception(exception: Exception, caused_callable: Any, args, kwargs):
        # See https://docs.python.org/3/library/inspect.html for more information on the inspect module.
        assert callable(caused_callable), 'Expected callable'
        if inspect.ismethod(caused_callable):
            func = dict(inspect.getmembers(caused_callable))['__func__']
            code = dict(inspect.getmembers(func))['__code__']
            callable_name = caused_callable.__qualname__
        elif inspect.isfunction(caused_callable):
            code = dict(inspect.getmembers(caused_callable))['__code__']
            callable_name = caused_callable.__qualname__
        else:
            # an object with __call__ method
            func = dict(inspect.getmembers(caused_callable.__call__))['__func__']
            code = dict(inspect.getmembers(func))['__code__']
            callable_name = caused_callable.__class__.__qualname__

        filename, lineno = code.co_filename, code.co_firstlineno
        raise type(exception)(
            f'{exception}, occurred when calling {callable_name} with \n'
            f'    args: {args}\n'
            f'  kwargs: {kwargs}\n'
            f'{callable_name} is defined at\n'
            f'  File "{filename}", line {lineno}'
        )

    def forward(self, *args):
        # pylint: disable=broad-except
        def load_arg(a, env):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        logger.info('start to interpret graph')

        args_iter = iter(args)
        hidet_env: Dict[str, Any] = {}

        graph_hidet_output: Optional[Any] = None

        for idx, node in enumerate(self.graph.nodes):
            assert isinstance(node, torch.fx.Node)
            logger.debug(f"interpreting node {idx}: {node.format_node()}")

            if node.op == "placeholder":
                arg = next(args_iter)
                if isinstance(arg, torch.Tensor):
                    raise RuntimeError('input tensor must be hidet Tensor, got torch.Tensor')
                hidet_env[node.name] = arg
            elif node.op == "get_attr":
                target_atoms = node.target.split(".")
                attr = self.graph_module
                for i, atom in enumerate(target_atoms):
                    if not hasattr(attr, atom):
                        raise RuntimeError(f"Node referenced nonexistent target {target_atoms[:i]} not")
                    attr = getattr(attr, atom)
                hidet_env[node.name] = tensor_from_torch(attr) if isinstance(attr, torch.Tensor) else attr
            elif node.op == "call_function":
                hidet_func = Registry.registered_functions[node.target]
                hidet_args = load_arg(node.args, hidet_env)
                hidet_kwargs = load_arg(node.kwargs, hidet_env)
                try:
                    hidet_env[node.name] = hidet_func(*hidet_args, **hidet_kwargs)
                except Exception as e:
                    self._raise_exception(e, hidet_func, hidet_args, hidet_kwargs)
            elif node.op == "call_method":
                args = load_arg(node.args, hidet_env)
                kwargs = load_arg(node.kwargs, hidet_env)

                if isinstance(args[0], Tensor):
                    torch_method = getattr(torch.Tensor, node.target)
                else:
                    torch_method = getattr(type(args[0]), node.target)
                hidet_method = self._lookup_hidet_method(torch_method)
                try:
                    hidet_env[node.name] = hidet_method(*args, **kwargs)
                except Exception as e:
                    self._raise_exception(e, hidet_method, args, kwargs)
            elif node.op == "call_module":
                hidet_module = self._lookup_hidet_module(node.target)
                args = load_arg(node.args, hidet_env)
                kwargs = load_arg(node.kwargs, hidet_env)
                try:
                    hidet_env[node.name] = hidet_module(*args, **kwargs)
                except Exception as e:
                    self._raise_exception(e, hidet_module, args, kwargs)
            elif node.op == "output":
                graph_hidet_output = hidet_env[node.name] = load_arg(node.args[0], hidet_env)
            else:
                assert False

        logger.info('finish interpreting graph')

        warnings.reset()

        return graph_hidet_output

    def forward_with_check(self, *args) -> str:
        # pylint: disable=broad-except
        def to_hidet(value):
            if isinstance(value, torch.Tensor):
                return tensor_from_torch(value.clone())
            return value

        def to_torch(value):
            if isinstance(value, Tensor):
                if value.is_symbolic():
                    raise ValueError('expect concrete arguments to check the correctness')
                value = value.torch()
            return value

        def load_arg(a, env):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        logger.info('start to interpret graph')

        args_iter = iter(args)
        torch_env: Dict[str, Any] = {}
        hidet_env: Dict[str, Any] = {}
        check_report: List[Tuple[str, str, str, str, str]] = [('kind', 'operator', 'dtype', 'error', 'attention')]

        for idx, node in enumerate(self.graph.nodes):
            assert isinstance(node, torch.fx.Node)
            logger.debug(f"interpreting node {idx}: {node.format_node()}")

            readable_target = ''

            if node.op == "placeholder":
                arg = next(args_iter)
                torch_env[node.name] = to_torch(arg)
                hidet_env[node.name] = to_hidet(arg)
            elif node.op == "get_attr":
                target_atoms = node.target.split(".")
                attr = self.graph_module
                for i, atom in enumerate(target_atoms):
                    if not hasattr(attr, atom):
                        raise RuntimeError(f"Node referenced nonexistent target {target_atoms[:i]} not")
                    attr = getattr(attr, atom)
                torch_env[node.name] = attr
                hidet_env[node.name] = to_hidet(attr)
            elif node.op == "call_function":
                torch_func = node.target
                torch_args = load_arg(node.args, torch_env)
                torch_kwargs = load_arg(node.kwargs, torch_env)
                torch_env[node.name] = torch_func(*torch_args, **torch_kwargs)

                hidet_func = Registry.registered_functions[torch_func]
                hidet_args = load_arg(node.args, hidet_env)
                hidet_kwargs = load_arg(node.kwargs, hidet_env)

                try:
                    hidet_env[node.name] = hidet_func(*hidet_args, **hidet_kwargs)
                except Exception as e:
                    self._raise_exception(e, hidet_func, hidet_args, hidet_kwargs)

                readable_target = self._get_callable_name(torch_func)
            elif node.op == "call_method":
                torch_args = load_arg(node.args, torch_env)
                torch_kwargs = load_arg(node.kwargs, torch_env)
                torch_method = getattr(type(torch_args[0]), node.target)
                torch_env[node.name] = torch_method(*torch_args, **torch_kwargs)

                hidet_args = load_arg(node.args, hidet_env)
                hidet_kwargs = load_arg(node.kwargs, hidet_env)
                hidet_method = self._lookup_hidet_method(torch_method)

                try:
                    hidet_env[node.name] = hidet_method(*hidet_args, **hidet_kwargs)
                except Exception as e:
                    self._raise_exception(e, hidet_method, hidet_args, hidet_kwargs)

                readable_target = self._get_callable_name(torch_method)
            elif node.op == "call_module":
                torch_module = self.torch_modules[node.target]
                torch_args = load_arg(node.args, torch_env)
                torch_kwargs = load_arg(node.kwargs, torch_env)
                try:
                    torch_env[node.name] = torch_module(*torch_args, **torch_kwargs)
                except Exception as e:
                    self._raise_exception(e, torch_module, torch_args, torch_kwargs)

                hidet_module = self._lookup_hidet_module(node.target)
                hidet_args = load_arg(node.args, hidet_env)
                hidet_kwargs = load_arg(node.kwargs, hidet_env)

                try:
                    hidet_env[node.name] = hidet_module(*hidet_args, **hidet_kwargs)
                except Exception as e:
                    self._raise_exception(e, hidet_module, hidet_args, hidet_kwargs)

                readable_target = self._get_callable_name(torch_module)
            elif node.op == "output":
                torch_env[node.name] = load_arg(node.args[0], torch_env)
                hidet_env[node.name] = load_arg(node.args[0], hidet_env)
            else:
                assert False

            torch_output = torch_env[node.name]
            hidet_output = hidet_env[node.name]

            if isinstance(torch_output, (list, tuple)) and isinstance(torch_output[0], torch.Tensor):
                torch_output = torch_output[0]
                hidet_output = hidet_output[0]

            if isinstance(torch_output, torch.Tensor) and isinstance(hidet_output, Tensor):
                error = relative_absolute_error(actual=torch_output, expected=hidet_output.torch())
                dtype = data_type(hidet_output.dtype)
                if dtype.is_integer():
                    pay_attention = error > 1e-8  # int32, ...
                elif dtype.is_float():
                    if dtype.nbytes <= 2:
                        pay_attention = error > 1e-1  # fp16
                    else:
                        pay_attention = error > 5e-5  # fp32
                else:
                    pay_attention = False
                check_report.append(
                    (node.op, readable_target, dtype.name, f'{error:.1e}', '<------' if pay_attention else '')
                )

        logger.info('finish interpreting graph')

        warnings.reset()

        return tabulate.tabulate(
            tabular_data=check_report[1:],
            headers=check_report[0],
            floatfmt='.3e',
            showindex=True,
            disable_numparse=True,
        )
