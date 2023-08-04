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

from typing import List, Dict
from itertools import product, chain

from hidet.graph import Operator
from hidet.ir import TensorElement, Var, Expr
from hidet.ir.compute import GridCompute, ReduceCompute, ArgReduceCompute, TensorInput
from hidet.ir.functors import ExprRewriter, ComputeRewriter
from hidet.ir.analyzers.bound_analyzer import infer_bound, BoundInfo


from .shard import TensorShardSpec, OpShardSpec, get_tile, AllReduce, ReduceScatter


class IndexRewriter(ExprRewriter, ComputeRewriter):
    def __init__(self):
        super().__init__()
        self.var2idx: Dict[Var, Expr] = {}
        self.input_accessed_indices: Dict[TensorInput, List[List[Expr]]] = {}
        self.reduce_axis_shapes: Dict[Var, Expr] = {}

    def visit_Var(self, e: Var):
        if e in self.var2idx:
            return self.var2idx[e]
        else:
            return super().visit_Var(e)

    def visit_TensorElement(self, e: TensorElement):
        tensor = e.base
        indices = tuple((self.visit(idx) for idx in e.indices))
        if isinstance(tensor, GridCompute):
            # passing the indices to the dependending grids's axes
            for axis, index in zip(tensor.axes, indices):
                self.var2idx[axis] = index
        elif isinstance(tensor, TensorInput):
            if tensor not in self.input_accessed_indices:
                self.input_accessed_indices[tensor] = []
            self.input_accessed_indices[tensor].append(indices)
        else:
            raise ValueError()
        tensor = self.visit(tensor)
        return TensorElement(tensor, indices)

    def visit_ReduceCompute(self, node: ReduceCompute):
        for axis, axis_len in zip(node.axes, node.shape):
            self.reduce_axis_shapes[axis] = axis_len
        return super().visit_ReduceCompute(node)

    def visit_ArgReduceCompute(self, node: ArgReduceCompute):
        self.reduce_axis_shapes[node.axis] = node.extent
        return super().visit_ArgReduceCompute(node)


class DataDependencyAnalyzer:
    def __init__(self, op: Operator, num_shards: int):
        self.op: Operator = op
        self.num_shards: int = num_shards

        index_rewriter = IndexRewriter()
        index_rewriter.visit(self.op.task.outputs)
        self.valid = all(isinstance(out, GridCompute) for out in self.op.task.outputs)
        self.input_accessed_indices = index_rewriter.input_accessed_indices
        self.reduce_axis_shapes = index_rewriter.reduce_axis_shapes

    def check(self, in_shard_dim: List[int], out_shard_dim: List[int], shard_reduce: bool = False):
        # Now only supports 1D partition
        task = self.op.task

        def _bound_info(rank, shape, is_shard=False):
            shape = int(shape)
            if is_shard:
                shard_size = shape // self.num_shards
                return BoundInfo(min_value=rank * shard_size, max_value=(rank + 1) * shard_size - 1)
            else:
                return BoundInfo(min_value=0, max_value=shape - 1)

        def _check(sharded_reduce_axes=None):
            if sharded_reduce_axes is None:
                sharded_reduce_axes = []
            # Enumerate all ranks
            for rank in range(self.num_shards):
                out_and_reduce_bound = {}
                # Generate the premises of output indices
                for o, shard_dim in zip(task.outputs, out_shard_dim):
                    if not isinstance(o, GridCompute):
                        return False
                    for i, (axis, shape) in enumerate(zip(o.axes, o.shape)):
                        out_and_reduce_bound[axis] = _bound_info(rank, shape, i == shard_dim)

                    for axis, shape in self.reduce_axis_shapes.items():
                        out_and_reduce_bound[axis] = _bound_info(rank, shape, axis in sharded_reduce_axes)
                        # here we assume reduction axis won't be reused

                for input_tensor, accessed_indices in self.input_accessed_indices.items():
                    shard_dim = in_shard_dim[task.inputs.index(input_tensor)]
                    if shard_dim is None:
                        continue
                    for indices in accessed_indices:
                        idx = indices[shard_dim]
                        shape = int(input_tensor.shape[shard_dim])
                        shard_size = shape // self.num_shards
                        lower, upper = rank * shard_size, (rank + 1) * shard_size
                        bound = infer_bound(idx, out_and_reduce_bound)[idx]
                        if bound.possible_min_value() is None or bound.possible_max_value() is None:
                            return False
                        if bound.possible_min_value() < lower or bound.possible_max_value() >= upper:
                            return False
            return True

        if shard_reduce:
            # We only analyze sharding reduction axis for single-output op
            if len(out_shard_dim) > 1:
                return False

            o = task.outputs[0]
            # We only consider the outer most reduction axis
            if not isinstance(o.value, ReduceCompute):
                return False

            if str(o.value.reduce_operation) not in ('sum', 'prod', 'max', 'min', 'avg'):
                # other reduce ops are not supported by NCCL
                return False

            # Try reduction on each axis to see if the result can be fixed.
            for shard_axis in o.value.axes:
                if _check([shard_axis]):
                    return True
            return False
        else:
            return _check()




def op_shard_rule_search(op: Operator, num_shards: int) -> List[OpShardSpec]:
    # Now we only search for 1D partition
    inputs = op.inputs
    outputs = op.outputs
    found_rules = []

    # We do not allow dynamic shapes
    if len(op.task.symbols) > 0:
        raise NotImplementedError("Dynamic shapes are not supported now.")

    # num_shards == 1 will break following logic
    if num_shards == 1:
        return [
            OpShardSpec(
                [TensorShardSpec(len(t.shape)) for t in inputs], [TensorShardSpec(len(t.shape)) for t in outputs]
            )
        ]

    # Initialize data dependency analyzer
    # If it is not valid, it means we meet some scenarios we cannot analyze
    # And we won't shard this op
    data_dependency_analyzer = DataDependencyAnalyzer(op, num_shards)
    if not data_dependency_analyzer.valid:
        return []

    # enumerate all possible input shardings. -1 means duplicate
    for in_shard_dims in product(*(chain([None], range(len(i.shape))) for i in inputs)):
        # Get a tile of each input tensor according to the input sharding
        # And compute the output shape
        try:
            new_inputs = []
            for i, shard_dim in enumerate(in_shard_dims):
                if shard_dim is not None:
                    if inputs[i].shape[shard_dim] % num_shards != 0:
                        break
                    new_inputs.append(get_tile(inputs[i], shard_dim, num_shards))
                else:
                    new_inputs.append(inputs[i])
            if len(new_inputs) < len(inputs):
                continue  # Illegal sharding, skip
            outputs = op.reforward(new_inputs)
        except (ValueError, RuntimeError, AssertionError, IndexError):
            continue

        # find the sharded output dimension
        # for each output tensor, there should be at most one dimension being sharded
        out_shard_dims = []
        for i, out in enumerate(outputs):
            origin_shape = list(op.outputs[i].shape)
            if list(out.shape) == origin_shape:
                out_shard_dims.append(None)
                continue
            shard_dim = None
            # None means invalid or not found here, since 'not to shard' is handled above
            for dim in range(len(out.shape)):
                if out.shape[dim] == origin_shape[dim] // num_shards:
                    if shard_dim is None:
                        shard_dim = dim
                    else:  # which means we have already found sharded dim
                        shard_dim = None
                        break  # Because there should be at most one sharded dimension
                elif out.shape[dim] != origin_shape[dim]:
                    # there should not be any shape other than unsharded and sharded
                    shard_dim = None
                    break
            if shard_dim is None:
                break
            # output is a valid result of a sharding
            out_shard_dims.append(shard_dim)

        if len(out_shard_dims) == len(outputs):  # shape analysis valid
            # data dependency analysis
            in_specs = [TensorShardSpec(len(i.shape), shard_dim) for i, shard_dim in zip(inputs, in_shard_dims)]
            out_specs = [TensorShardSpec(len(o.shape), shard_dim) for o, shard_dim in zip(outputs, out_shard_dims)]
            if data_dependency_analyzer.check(in_shard_dims, out_shard_dims):
                found_rules.append(OpShardSpec(in_specs, out_specs))
            else:  # Failed, but it still can be partial results. Try to fix it with reduction.
                if data_dependency_analyzer.check(in_shard_dims, out_shard_dims, shard_reduce=True):
                    reduce_op = op.task.outputs[0].value.reduce_operation
                    found_rules.append(OpShardSpec(in_specs, out_specs, reduce_fn=[AllReduce(reduce_op)]))
                    # if all_reduce is valid, then reduce_scatter is also valid
                    output = op.task.outputs[0]
                    if output.shape[0] % num_shards == 0:
                        out_spec = TensorShardSpec(len(output.shape), 0)
                        found_rules.append(OpShardSpec(in_specs, [out_spec], reduce_fn=[ReduceScatter(reduce_op)]))

    return found_rules
