"""
Add PyTorch Operator Mapping
============================

This guide describes how to add an operator mapping for PyTorch.

.. graphviz::
  :caption: The workflow of hidet backend of :code:`torch.compile(..., backend='hidet')`.

  digraph {
      // rankdir=LR;
      splines=curved;
      node [
          shape=box, style="rounded, filled",
          height=0.4, width=0.6, margin="0.2,0.10",
          fillcolor="#EEF0E5",
          color="#163020",
          fontcolor="#163020",
      ];
      edge [
          color="#163020",
          fontcolor="#163020",
      ];


      graph [style="rounded, dashed"]
          a [label="PyTorch nn.Module"];
          b [label="torch.fx.Graph"];
          c [label="hidet.FlowGraph"];
          d [label="hidet.runtime.CompiledGraph"];

          a -> b [label="   Step 1: PyTorch Dynamo"];
          b -> c [label="   Step 2: Operator mapping"];
          c -> d [label="   Step 3: FlowGraph building"];
  }

During step 2, we convert each pytorch operator to a hidet operator. In a `torch.fx.Graph`, there are three kinds of
operators that need to be converted:

1. functions (e.g., :code:`torch.nn.functional.relu`, :code:`torch.relu`, :code:`operator.add`, etc.)
2. modules (e.g., :code:`torch.nn.ReLU`, :code:`torch.nn.Linear`, etc.)
3. tensor methods (e.g., :code:`torch.Tensor.squeeze`, :code:`torch.Tensor.to`, etc.)

In this guide, we will show how to add the operator mapping for all the three kinds of operators.

1. Prepare Environment
----------------------
First, we remove some existing operator mapping (i.e., conversion) rules for demonstration purpose, and define an
example model.
"""
import operator
import torch
from torch import nn

# hidet employs an interpreter to convert a fx.Graph to FlowGraph
from hidet.graph.frontend.torch.registry import Registry

# the following three modules register the conversion rules
import hidet.graph.frontend.torch.register_functions
import hidet.graph.frontend.torch.register_modules
import hidet.graph.frontend.torch.register_methods

# Before removing registered functions, make sure to
# call allow_in_graph_registered_funcs_only() by importing dynamo_backends
import hidet.graph.frontend.torch.dynamo_backends

# we remove the rules for the following operators for demonstration purpose
# we will add them back later
del Registry.registered_functions[torch.nn.functional.relu]
del Registry.registered_functions[operator.add]
del Registry.registered_modules[torch.nn.Linear]
del Registry.registered_methods[torch.Tensor.flatten]


class Model(nn.Module):
    """a model used nn.Linear, nn.functional.relu, operator.add and Tensor.flatten"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.relu(x)
        x = x + x
        return x.flatten()


# %%
# 2. Compile and Run the Model
# ----------------------------
# If we compile and run the model, we will get an error that complains about the missing conversion rules for
# :code:`torch.nn.Linear`, :code:`torch.nn.functional.relu` and :code:`operator.add`.


def run_model():
    model = Model().cuda()
    model_opt = torch.compile(model, backend='hidet', mode='max-autotune')

    x = torch.randn(10, 10, device='cuda')
    y1 = model_opt(x)
    y2 = model(x)
    torch.testing.assert_close(actual=y1, expected=y2, atol=3e-3, rtol=3e-3)
    print('success!')


try:
    run_model()
except Exception as e:
    print(e)

# %%
# 3. Add Operator Mappings
# ------------------------
#
from typing import Optional
from hidet import ops
from hidet import Tensor
from hidet.graph.frontend.torch.registry import (
    register_function,
    register_module,
    register_method,
    HidetModule,
)


# register the conversion rule for torch.nn.functional.relu
@register_function(torch.nn.functional.relu)
def torch_relu(x: Tensor, inplace: bool = False):  # the signature must match the original function
    # the parameter `x` is hidet.Tensor instead of torch.Tensor
    # we also need to return a hidet.Tensor instead of torch.Tensor
    _ = inplace  # ignore inplace
    return ops.relu(x)


@register_function(operator.add)
def operator_add(x: Tensor, y: Tensor):
    return ops.add(x, y)


@register_module(torch.nn.Linear)
class HidetLinear(
    HidetModule
):  # HidetModule is a tool class that helps us to convert a torch.nn.Module
    def __init__(self, torch_module: torch.nn.Module):
        super().__init__(torch_module)
        # inside the class, we can access the parameter of the torch module via
        # `self.param(name: str, optional: bool = False) -> Tensor`
        # and the returned tensor is a hidet.Tensor
        self.transposed_weight: Tensor = ops.transpose(self.param('weight'), [1, 0])
        self.bias: Optional[Tensor] = self.param('bias', optional=True)

    def __call__(self, x: Tensor) -> Tensor:
        # similarly, the parameter `x` is hidet.Tensor instead of torch.Tensor
        y = ops.matmul(x, self.transposed_weight)
        if self.bias is not None:
            y = y + self.bias
        return y


# %%
# If we run the model again, it will complain about the missing conversion rule for :code:`torch.Tensor.flatten`.
# It does not complain about missing conversion rule for :code:`torch.Tensor.flatten` before because we can not
# know the type of the method's class (i.e., :code:`torch.Tensor`) before we actually run the model.
#
try:
    run_model()
except Exception as e:
    print(e)


# %%
# Thus, we need to add the conversion rule for :code:`torch.Tensor.flatten` later as well.


@register_method(torch.Tensor.flatten)
def tensor_flatten(self: Tensor, start_dim=0, end_dim=-1):
    return ops.flatten(self, start_dim=start_dim, end_dim=end_dim)


run_model()

# %%
# We put all the registration code in the following three modules:
#
# 1. :code:`hidet.graph.frontend.torch.register_functions` (all the functions in `torch.nn.functional.*` and
#    `operator.*`)
# 2. :code:`hidet.graph.frontend.torch.register_modules` (all the modules in `torch.nn.*`)
# 3. :code:`hidet.graph.frontend.torch.register_methods` (all the methods in `torch.Tensor.*`)
#
# Lots of operators have already been registered in the above three modules, and they are also good examples for us
# to learn how to add operator mapping.
#
# Usually, we will use the existing operators in hidet (defined in `hidet.ops.*`) to implement the pytorch operators.
# If there are no corresponding operators in hidet, we can add the missing operators to `hidet.ops.*` by following the
# guide :doc:`/how-to-guides/add-new-operator/index`.
#
# .. note::
#    The operator mapping rules are registered in the global registry. Thus, if we register the same operator mapping
#    rules multiple times, only the last registration will take effect.

# %%
# 4. Summary
# ----------
# In this guide, we show how to add operator mapping for PyTorch. We first remove some existing operator mapping rules
# for demonstration purpose, and then add them back. We also show how to add operator mapping for functions, modules
# and tensor methods.
#
