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
from typing import Union, List, Dict

from hidet.ir.expr import Var, Expr
from hidet.ir.functors import IRVisitor

from hidet.ir.cute.expr import CallOp

from hidet.ir.func import Function
from hidet.ir.stmt import DeclareStmt
from hidet.ir.tools import TypeInfer, infer_type

from hidet.ir.cute import TensorLayout, make_layout
from hidet.ir.cute.ops import (
    PartitionSrc,
    PartitionDst,
    SubTensor,
    TensorBase,
    Transpose,
    PartitionA,
    PartitionB,
    MBarriers,
)


class TensorInfo:
    """
    A class to track the information of a tensor, including its dimensions and layout.

    Note: the dimension attribute is used when there is `Transpose` operator in the IR.
    Basically, we allow user to transpose the view of a tensor in the shared memory. For
    example, we can have the following IR sequence:
    ```python
    a = tensor("float16", [128, 128], "shared")
    b = transpose(a, [1, 0]) # b is an alias of a with transposed view.
    ```
    In this case, the `dims` attribute of `TensorInfo` object for `b` will be `[1, 0]`.

    Attributes:
        tensor (TensorBase): The tensor object.
        dims (List[int], optional): The dimensions of the tensor. Default is None.

    Methods:
        tensor() -> TensorBase:
            Returns the tensor object.

        dims() -> List[int]:
            Returns the dimensions of the tensor.

        layout() -> TensorLayout:
            Returns the layout of the tensor based on the specified dimensions.

        set_dims(dims: List[int]):
            Sets the dimensions of the tensor.

        __str__() -> str:
            Returns a string representation of the TensorInfo object.
    """

    def __init__(self, tensor: Union[TensorBase, MBarriers], dims: List[int] = None):
        """
        Initializes a TensorInfo object.

        Args:
            tensor (TensorBase): The tensor object.
            dims (List[int], optional): The dimensions of the tensor. Default is None.
        """
        self._tensor: Union[TensorBase, MBarriers] = tensor
        self._dims: List[int] = dims

    @property
    def tensor(self) -> Union[TensorBase, MBarriers]:
        """Returns the tensor object."""
        return self._tensor

    @property
    def dims(self):
        """Returns the dimensions of the tensor."""
        return self._dims

    @property
    def layout(self) -> TensorLayout:
        """
        Returns the layout of the tensor based on the specified dimensions.
        If dimensions are not specified, returns the original layout of the tensor.

        Returns:
            TensorLayout: The layout of the tensor.
        """
        ty = infer_type(self.tensor.make_call())
        layout = ty.layout

        if self._dims is None:
            return layout

        modes = [layout[d] for d in self.dims]
        return make_layout(*modes)

    def set_dims(self, dims: List[int]):
        """
        Sets the dimensions of the tensor.

        Args:
            dims (List[int]): The dimensions to set.
        """
        self._dims = dims

    def __str__(self):
        """
        Returns a string representation of the TensorInfo object.

        Returns:
            str: String representation of the TensorInfo object.
        """
        return f"{{tensor:{self.tensor}, layout:{self.layout}}}"


def tensor_info(tensor: Union[TensorBase, MBarriers], *dims):
    """
    Creates and returns a TensorInfo object based on the provided tensor and optional dimensions.

    Args:
        tensor: Union[TensorBase, MBarriers]: The tensor object.
        dims: Optional dimensions for the tensor.

    Returns:
        TensorInfo: The created TensorInfo object.
    """
    if len(dims) == 0:
        return TensorInfo(tensor)
    else:
        return TensorInfo(tensor, dims)


class TensorAliasAnalysis(IRVisitor):
    """
    A class to analyze tensor aliases.

    In the Cute IR, a tensor can only be created by `make_tensor` or `tensor_view` operations within a declare
    statement, and it can then be partitioned, sliced or transposed. All the subtensors or tensor views created
    by the above operations are considered as aliases of the tensor.

    Attributes:
        var2tensor (Dict[Var, TensorInfo]): Mapping of variables to their corresponding tensor information.
        var2var (Dict[Expr, Expr]): Mapping of variable relationships.

    Methods:
        get_tensor(v: Var) -> TensorInfo:
            Resolves and returns the parent tensor for a given variable.

        visit_DeclareStmt(stmt: DeclareStmt):
            Processes declaration statements to extract tensor-related information and update the mappings.

        analyze(func: Function) -> Dict[Var, TensorInfo]:
            Initiates the analysis for a given function and returns the mapping of variables to their tensor
            information.
    """

    def __init__(self):
        """Initializes a TensorAliasAnalysis object."""

        super().__init__()
        self.var2tensor: Dict[Var, TensorInfo] = {}
        self.var2var: Dict[Expr, Expr] = {}
        self.infer_type = TypeInfer()

    def get_tensor(self, v: Var):
        """
        Resolves and returns the parent tensor for a given variable by following variable mappings.

        Args:
            v (Var): The variable for which to resolve the parent tensor.

        Returns:
            TensorInfo: The resolved parent tensor information.
        """
        assert v in self.var2var
        parent = self.var2var[v]
        while parent in self.var2var:
            parent = self.var2var[parent]
        if parent in self.var2tensor:
            return self.var2tensor[parent]
        else:
            return None

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        """
        Processes declaration statements to extract tensor-related information and update the mappings.

        Args:
            stmt (DeclareStmt): The declaration statement to process.
        """
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            v = stmt.var
            op = call.op
            self.visit(op)
            if isinstance(op, (TensorBase, MBarriers)):
                tensor = tensor_info(op)
                self.var2tensor[v] = tensor
            elif isinstance(op, (PartitionSrc, PartitionDst, SubTensor)):
                self.var2var[v] = op.x
                tensor = self.get_tensor(v)
                if tensor is not None:
                    self.var2tensor[v] = tensor
            elif isinstance(op, (PartitionA, PartitionB)):
                v_ty = self.infer_type(v)
                if v_ty.scope.is_shared():
                    self.var2var[v] = op.x
                    tensor = self.get_tensor(v)
                    if tensor is not None:
                        self.var2tensor[v] = tensor
            elif isinstance(op, Transpose):
                tensor = self.var2tensor[op.x]
                self.var2tensor[v] = tensor_info(tensor.tensor, *op.dims)

    def analyze(self, func: Function):
        """
        Initiates the analysis for a given function and returns the mapping of variables to their tensor information.

        Args:
            func (Function): The function to analyze.

        Returns:
            Dict[Var, TensorInfo]: The mapping of variables to their tensor information.
        """
        self.visit(func)

        return self.var2tensor
