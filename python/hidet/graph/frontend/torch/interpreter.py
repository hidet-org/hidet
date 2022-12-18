from __future__ import annotations
from typing import Dict, Any, Type, Callable, Optional, Tuple, Set
import operator
import warnings
import itertools
import torch
from torch.fx import Node

from hidet.graph.module import Module
from hidet.graph.tensor import Tensor
from hidet.graph.tensor import from_torch as tensor_from_torch
from .availability import available


class Registry:
    registered_modules: Dict[Type[torch.nn.Module], Type[HidetModule]] = {}
    registered_functions: Dict[Callable, Callable] = {}
    registered_methods: Dict[Tuple[Type, str], Callable] = {}


class ExpectedRegistry:
    operator_functions: Set[Callable] = set(c for c in operator.__dict__.values() if callable(c))
    torch_functional: Set[Callable] = set(c for c in torch.nn.functional.__dict__.values() if callable(c))
    torch_modules: Set[Type[torch.nn.Module]] = set(
        c for c in torch.nn.__dict__.values() if isinstance(c, type) and issubclass(c, torch.nn.Module)
    )
    torch_root_functions: Set[Callable] = set(c for c in torch.__dict__.values() if callable(c))


_warning_require_grad = True


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


def register_method(cls: Type[object], method_name: str):
    def decorator(hidet_func):
        Registry.registered_methods[(cls, method_name)] = hidet_func
        return hidet_func

    return decorator


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
                if torch_param.requires_grad:
                    global _warning_require_grad
                    if _warning_require_grad:
                        warnings.warn(
                            f"hidet: weight '{name}' in module {self.mod} has requires_grad=True."
                            " Please consider calling .requires_grad_(False) on the model to silence this "
                            "warning"
                        )
                        _warning_require_grad = False
                    torch_param = torch_param.detach()
                self.hidet_params[name] = tensor_from_torch(torch_param)
        return self.hidet_params[name]


class ImportedTorchModule(Module):
    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__()
        self.graph_module: torch.fx.GraphModule = graph_module
        self.graph: torch.fx.Graph = graph_module.graph
        self.torch_modules: Dict[str, torch.nn.Module] = dict(graph_module.named_modules())
        self.hidet_modules: Dict[str, HidetModule] = {}

        self._check_support()

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
                if target in ExpectedRegistry.torch_modules:
                    lines.append("  torch.nn.{}".format(target.__name__))
                elif target in ExpectedRegistry.torch_functional:
                    lines.append("  torch.nn.functional.{}".format(target.__name__))
                elif target in ExpectedRegistry.operator_functions:
                    lines.append("  operator.{}".format(target.__name__))
                elif target in ExpectedRegistry.torch_root_functions:
                    lines.append("  torch.{}".format(target.__name__))
                else:
                    lines.append("  {}".format(target))
            raise NotImplementedError("\n".join(lines))

    def _lookup_hidet_module(self, target: str) -> HidetModule:
        if target not in self.hidet_modules:
            torch_module = self.torch_modules[target]
            torch_cls = type(torch_module)
            hidet_cls = Registry.registered_modules[torch_cls]
            self.hidet_modules[target] = hidet_cls(torch_module)
        return self.hidet_modules[target]

    def _lookup_hidet_method(self, self_obj, target: str):
        cls = type(self_obj)
        if (cls, target) not in Registry.registered_methods:
            raise NotImplementedError(f"hidet: method {cls.__name__}.{target} is not supported yet.")
        return Registry.registered_methods[(cls, target)]

    def _access_attribute(self, target: str) -> Any:
        target_atoms = target.split(".")
        attr = self.graph_module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {target_atoms[:i]} not")
            attr = getattr(attr, atom)
        if isinstance(attr, torch.Tensor):
            if attr.requires_grad:
                global _warning_require_grad
                if _warning_require_grad:
                    warnings.warn(
                        f"hidet: weight '{target}' in module {self.mod} has requires_grad=True,"
                        " Please consider calling .requires_grad_(False) on the model to silence this "
                        "warning"
                    )
                    _warning_require_grad = False
                attr = attr.detach()
            attr = tensor_from_torch(attr)
        return attr

    @staticmethod
    def _reset_warning_flag():
        global _warning_require_grad
        _warning_require_grad = True  # reset warning flag

    def forward(self, *args):
        args_iter = iter(args)
        env: Dict[str, Any] = {}
        output = None

        def load_args(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        for node in self.graph.nodes:
            assert isinstance(node, Node)
            if node.op == "placeholder":
                env[node.name] = next(args_iter)
            elif node.op == "get_attr":
                env[node.name] = self._access_attribute(node.target)
            elif node.op == "call_function":
                func = Registry.registered_functions[node.target]
                env[node.name] = func(*load_args(node.args), **load_args(node.kwargs))
            elif node.op == "call_method":
                self_obj, *args = load_args(node.args)
                kwargs = load_args(node.kwargs)
                method = self._lookup_hidet_method(self_obj, node.target)
                env[node.name] = method(self_obj, *args, **kwargs)
            elif node.op == "call_module":
                hidet_module: HidetModule = self._lookup_hidet_module(node.target)
                self_obj, *args = load_args(node.args)
                kwargs = load_args(node.kwargs)
                env[node.name] = hidet_module(self_obj, *args, **kwargs)
            elif node.op == "output":
                output = node.args
            else:
                assert False

        self._reset_warning_flag()

        return load_args(output)


def from_torch(module):
    """
    Convert a torch.nn.Module or torch.fx.GraphModule to a hidet.nn.Module.

    Parameters
    ----------
    module: Union[torch.nn.Module, torch.fx.GraphModule]
        The torch module to convert.

    Returns
    -------
    ret: ImportedTorchModule
        The converted hidet module, which is a subclass of hidet.nn.Module.
    """
    if not available():
        raise RuntimeError('torch is not available.')

    if isinstance(module, torch.fx.GraphModule):
        graph_module = module
    elif isinstance(module, torch.nn.Module):
        graph_module = torch.fx.symbolic_trace(module)
    else:
        raise ValueError(f'Current only support import torch.nn.Module and torch.fx.GraphModule, got {type(module)}.')
    return ImportedTorchModule(graph_module)
