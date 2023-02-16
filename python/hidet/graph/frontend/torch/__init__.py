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
from .availability import available, dynamo_available, imported
from . import utils
from .dynamo_config import dynamo_config, DynamoConfig


def from_torch(module, concrete_args=None):
    """
    Convert a torch.nn.Module or torch.fx.GraphModule to a hidet.nn.Module.

    Parameters
    ----------
    module: torch.nn.Module or torch.fx.GraphModule
        The torch module to convert.

    concrete_args: Dict[str, Any] or None
        The concrete arguments to the module. If provided, will be used to make some arguments concrete during symbolic
        tracing.

    Returns
    -------
    ret: Interpreter
        The converted hidet module, which is a subclass of hidet.nn.Module.
    """
    import torch
    from . import register_functions, register_modules, register_methods  # pylint: disable=unused-import
    from .interpreter import Interpreter

    if not available():
        raise RuntimeError('torch is not available.')

    if isinstance(module, torch.fx.GraphModule):
        graph_module = module
    elif isinstance(module, torch.nn.Module):
        graph_module = torch.fx.symbolic_trace(module, concrete_args=concrete_args)
    else:
        raise ValueError(f'Current only support import torch.nn.Module and torch.fx.GraphModule, got {type(module)}.')
    return Interpreter(graph_module)


def register_dynamo_backends():
    print(
        'Now, hidet will use the entry_points mechanism to register as a dynamo backend. \n'
        'Feel free to remove the line `hidet.frontend.torch.register_dynamo_backends()` in your code.'
    )
