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

from typing import List
from itertools import product

import hidet
from hidet.graph import Operator
from hidet.ir import TensorElement, Var
from hidet.ir.tools import collect
from hidet.ir.compute import GridCompute, ReduceCompute, TensorInput
from hidet.ir.functors import ExprRewriter, ComputeRewriter
from hidet.ir.analyzers.bound_analyzer import infer_bound, BoundInfo
from hidet.graph.ops.matmul import MatmulOp

from .shard import TensorShardSpec, OpShardSpec, get_tile, AllReduce, ReduceScatter


class IndexRewriter(ExprRewriter, ComputeRewriter):
    def __init__(self):
        super().__init__()
        self.var2idx = None
        self.root = None

    def process_expr(self, e, var2idx):
        self.var2idx = var2idx
        self.root = e
        return self.visit(e)

    def visit_Var(self, e: Var):
        if e in self.var2idx:
            return self.var2idx[e]
        else:
            return super().visit_Var(e)

    def visit_GridCompute(self, node: GridCompute):
        if self.root is node:
            return super().visit_GridCompute(node)
        else:
            # do not go deeper
            return node


class DataDependencyAnalyzer:
    def __init__(self, op: Operator, num_shards: int):
        self.op = op
        self.num_shards = num_shards
        self.valid = self._build_bound_expr()

    def _build_bound_expr(self) -> bool:
        task = self.op.task
        outputs = task.outputs

        index_rewriter = IndexRewriter()

        # collect all reduction axes.
        reductions = set.union(*(set(collect(o, ReduceCompute)) for o in outputs))
        self.reduce_axis_shapes = {}
        for reduction in reductions:
            for axis, axis_len in zip(reduction.axes, reduction.shape):
                self.reduce_axis_shapes[axis] = axis_len
        self.input_accessed_indices = {}

        def _build_grid_bound(grid, axis_map):
            tensor_element_exprs = collect(grid, TensorElement, stop_when_found=True)
            for te in tensor_element_exprs:
                # note that one grid compute might appear in multiple items in a expression
                tensor = te.base
                if isinstance(tensor, GridCompute):
                    # passing the indices to the dependending grids's axes
                    new_axis_map = {}
                    new_axis_map.update(axis_map)
                    for axis, index in zip(tensor.axes, te.indices):
                        new_axis_map[axis] = index
                    tensor = index_rewriter.process_expr(tensor, new_axis_map)
                    _build_grid_bound(tensor, new_axis_map)
                elif isinstance(tensor, TensorInput):
                    if tensor not in self.input_accessed_indices:
                        self.input_accessed_indices[tensor] = []
                    self.input_accessed_indices[tensor].append(te.indices)

        for o in outputs:
            if not isinstance(o, GridCompute):
                return False  # we don't support scalar output
            _build_grid_bound(o, {})
        return True

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
                    if shard_dim < 0:
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
    assert len(op.task.symbols) == 0

    # num_shards == 1 will break following logic
    if num_shards == 1:
        return [
            OpShardSpec(
                [TensorShardSpec(len(t.shape)) for t in op.inputs], [TensorShardSpec(len(t.shape)) for t in op.outputs]
            )
        ]

    # Initialize data dependency analyzer
    # If it is not valid, it means we meet some scenarios we cannot analyze
    # And we won't shard this op
    data_dependency_analyzer = DataDependencyAnalyzer(op, num_shards)
    if not data_dependency_analyzer.valid:
        return []

    # enumerate all possible input shardings. -1 means duplicate
    for in_shard_dims in product(*(range(-1, len(i.shape)) for i in inputs)):
        if isinstance(op, MatmulOp):
            if all((in_shard_dim == -1 for in_shard_dim in in_shard_dims)):
                continue
        # Get a tile of each input tensor according to the input sharding
        # And compute the output shape
        try:
            new_inputs = []
            for i, shard_dim in enumerate(in_shard_dims):
                if shard_dim >= 0:
                    if inputs[i].shape[shard_dim] % num_shards != 0:
                        break
                    new_inputs.append(get_tile(inputs[i], num_shards, shard_dim))
                else:
                    new_inputs.append(inputs[i])
            if len(new_inputs) < len(inputs):
                continue
            outputs = op.reforward(new_inputs)
        except (ValueError, RuntimeError, AssertionError, IndexError):
            continue

        # find the sharded output dimension
        # for each output tensor, there should be at most one dimension being sharded
        out_shard_dims = []
        for i, out in enumerate(outputs):
            shard_dim = -1
            origin_shape = op.outputs[i].shape
            for dim in range(len(out.shape)):
                if out.shape[dim] == origin_shape[dim] // num_shards:
                    if shard_dim != -1:
                        shard_dim = None  # invalid output shape
                        # Because there should be at most one sharded dimension
                    else:
                        shard_dim = dim
                elif out.shape[dim] != origin_shape[dim]:
                    shard_dim = None  # invalid output shape
            if shard_dim is not None:  # output is invalid result of a sharding
                out_shard_dims.append(shard_dim)
            else:
                break

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


if __name__ == '__main__':
    import hidet.testing

    def resnet_test():
        model = hidet.testing.models.resnet.resnet18()
        x = hidet.symbol([8, 3, 224, 224])
        flow_graph = hidet.trace_from(model(x))

        cache = {}
        for op in flow_graph.nodes:
            print(op)
            if str(op) not in cache:
                shard_plans = op_shard_rule_search(op, 1)
                cache[str(op)] = shard_plans
            for sp in cache[str(op)]:
                print(sp)

    resnet_test()
