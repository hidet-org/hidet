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

from z3 import Int, And, Or, Not, Solver, IntVal, If

import hidet
from hidet.graph import Operator
from hidet.ir import TensorElement
from hidet.ir.tools import collect
from hidet.ir.compute import GridCompute, ReduceCompute, TensorInput
from hidet.ir.functors import ExprVisitor

from .shard import TensorShardSpec, OpShardSpec, ReduceFunction


class Z3ExprBuilder(ExprVisitor):
    # Translate Hidet expression to z3
    def __init__(self):
        super().__init__()
        self.var_table = {}

    def process_expr(self, e, var_table):
        self.var_table = var_table
        return self.visit(e)

    def visit_Var(self, e):
        assert e in self.var_table
        return self.var_table[e]

    def visit_Add(self, e):
        a, b = self.visit(e.a), self.visit(e.b)
        if a is None or b is None:
            return None
        return a + b

    def visit_Multiply(self, e):
        a, b = self.visit(e.a), self.visit(e.b)
        if a is None or b is None:
            return None
        return a * b

    def visit_Constant(self, e):
        val = e.value
        assert isinstance(val, int)
        return IntVal(val)

    def visit_Div(self, e):
        a, b = self.visit(e.a), self.visit(e.b)
        if a is None or b is None:
            return None
        return a / b

    def visit_Mod(self, e):
        a, b = self.visit(e.a), self.visit(e.b)
        if a is None or b is None:
            return None
        return a % b

    def visit_Sub(self, e):
        a, b = self.visit(e.a), self.visit(e.b)
        if a is None or b is None:
            return None
        return a - b

    def visit_LessThan(self, e):
        a, b = self.visit(e.a), self.visit(e.b)
        if a is None or b is None:
            return None
        return a < b

    def visit_LessEqual(self, e):
        a, b = self.visit(e.a), self.visit(e.b)
        if a is None or b is None:
            return None
        return a <= b

    def visit_Equal(self, e):
        a, b = self.visit(e.a), self.visit(e.b)
        if a is None or b is None:
            return None
        return a == b

    def visit_IfThenElse(self, e):
        cond, then_expr, else_expr = self.visit(e.cond), self.visit(e.then_expr), self.visit(e.else_expr)
        if cond is None or then_expr is None or else_expr is None:
            return None
        return If(cond, then_expr, else_expr)


class DataDependencyAnalyzer:
    def __init__(self, op: Operator, num_shards: int, symbol_table: dict = None):
        self.op = op
        self.num_shards = num_shards
        self.symbol_table = {} if symbol_table is None else symbol_table
        self.valid = self._build_bound_expr()

    def _build_bound_expr(self) -> bool:
        task = self.op.task
        outputs = task.outputs
        index_builder = Z3ExprBuilder()

        # collect all reduction axes.
        reductions = set.union(*(set(collect(o, ReduceCompute)) for o in outputs))
        self.reduce_axis_shapes = {}
        self.reduce_z3_vars = {}
        for reduction in reductions:
            for i, (axis, axis_len) in enumerate(zip(reduction.axes, reduction.shape)):
                self.reduce_axis_shapes[axis] = axis_len
                self.reduce_z3_vars[axis] = Int(f'r_{i}')
        self.input_boundaries = {}

        def _build_grid_bound(grid, axis_map):
            tensor_element_exprs = collect(grid, TensorElement, stop_when_found=True)
            for te in tensor_element_exprs:
                # note that one grid compute might appear in multiple items in a expression
                tensor = te.base
                indices_z3 = [index_builder.process_expr(idx, axis_map) for idx in te.indices]
                if isinstance(tensor, GridCompute):
                    # passing the indices to the dependending grids's axes
                    new_axis_map = {}
                    new_axis_map.update(self.symbol_table)
                    new_axis_map.update(axis_map)
                    for axis, index_z3 in zip(tensor.axes, indices_z3):
                        new_axis_map[axis] = index_z3
                    _build_grid_bound(tensor, new_axis_map)
                elif isinstance(tensor, TensorInput):
                    if tensor not in self.input_boundaries:
                        self.input_boundaries[tensor] = []
                    self.input_boundaries[tensor].append(indices_z3)

        self.grid_z3_vars = {}
        for o in outputs:
            if not isinstance(o, GridCompute):
                return False  # we don't support scalar output
            axis_map = {}
            for axis in o.axes:
                axis_map[axis] = Int(f'i_{len(self.grid_z3_vars)}')
                self.grid_z3_vars[axis] = axis_map[axis]
            axis_map.update(self.symbol_table)
            axis_map.update(self.reduce_z3_vars)
            _build_grid_bound(o, axis_map)
        return True

    def check(self, in_shard_dim: List[int], out_shard_dim: List[int], shard_reduce: bool = False):
        # Now only supports 1D partition
        task = self.op.task
        rank_z3 = Int('rank')
        rank_constraint = And(rank_z3 >= 0, rank_z3 < self.num_shards)
        out_and_reduce_constraints = []
        index_builder = Z3ExprBuilder()

        def _boundary_constraint(axis_z3, shape, sharded):
            # auxiliary function to generate boundary constraints
            if not isinstance(shape, int):
                shape = index_builder.process_expr(shape, self.symbol_table)
                shard_size = shape / self.num_shards
            else:
                shard_size = shape // self.num_shards
            if sharded:
                return And(axis_z3 >= rank_z3 * shard_size, axis_z3 < (rank_z3 + 1) * shard_size)
            else:
                return And(axis_z3 >= 0, axis_z3 < shape)

        # Generate the premises of output indices
        for o, shard_dim in zip(task.outputs, out_shard_dim):
            if not isinstance(o, GridCompute):
                return False
            for i, (axis, shape) in enumerate(zip(o.axes, o.shape)):
                axis_z3 = self.grid_z3_vars[axis]
                out_and_reduce_constraints.append(_boundary_constraint(axis_z3, shape, i == shard_dim))

            for axis, axis_z3 in self.reduce_z3_vars.items():
                shape = self.reduce_axis_shapes[axis]
                out_and_reduce_constraints.append(_boundary_constraint(axis_z3, shape, False))
                # here we assume reduction axis won't be reused

        # Generate the constraints of inputs indices (Is any input index out of boundary?)
        in_constraints = []
        for inp, shard_dim in zip(task.inputs, in_shard_dim):
            if inp not in self.input_boundaries:
                continue
            for indices in self.input_boundaries[inp]:
                for i, (axis_z3, shape) in enumerate(zip(indices, inp.shape)):
                    if i == shard_dim:
                        in_constraints.append(Not(_boundary_constraint(axis_z3, shape, True)))

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
                axis_z3 = self.reduce_z3_vars[shard_axis]
                shape = self.reduce_axis_shapes[shard_axis]
                c = _boundary_constraint(axis_z3, shape, True)
                if self._check(rank_constraint, in_constraints, out_and_reduce_constraints + [c]):
                    return True
            return False
        else:
            return self._check(rank_constraint, in_constraints, out_and_reduce_constraints)

    @staticmethod
    def _check(rank_constraint, in_constraints, out_and_reduce_constraints):
        constraints = [rank_constraint, Or(*in_constraints)] + out_and_reduce_constraints
        s = Solver()
        return str(s.check(*constraints)) == 'unsat'


def _get_first_tile(tensor, nshards, shard_dim):
    # Only works for 1-D partition
    slice_list = []
    for i in range(len(tensor.shape)):
        if i != shard_dim:
            slice_list.append(slice(tensor.shape[i]))
        else:
            shard_size = tensor.shape[i] // nshards
            slice_list.append(slice(shard_size))
    return tensor[slice_list]


def op_shard_rule_search(op: Operator, num_shards: int) -> List[OpShardSpec]:
    # Now we only search for 1D partition
    inputs = op.inputs
    outputs = op.outputs
    found_rules = []

    # Initialize the mapping from dynamic shape symbol to Z3 symbol
    symbols = op.task.symbols
    symbol_table = {sym: Int(sym.name) for sym in symbols}
    symbol_positive = [sym >= num_shards for sym in symbol_table.values()]

    # Initialize data dependency analyzer
    # If it is not valid, it means we meet some scenarios we cannot analyze
    # And we won't shard this op
    data_dependency_analyzer = DataDependencyAnalyzer(op, num_shards, symbol_table)
    if not data_dependency_analyzer.valid:
        return []

    # auxiliary function to check two values that might involve symbols
    def _symbol_eq(a, b):
        if isinstance(a, int) and isinstance(b, int):
            return a == b

        if str(a) == str(b):  # shortcut for identical expressions
            return True

        expr_builder = Z3ExprBuilder()
        if isinstance(a, int):
            a_z3 = IntVal(a)
        else:
            a_z3 = expr_builder.process_expr(a, symbol_table)

        if isinstance(b, int):
            b_z3 = IntVal(b)
        else:
            b_z3 = expr_builder.process_expr(b, symbol_table)

        neq = a_z3 != b_z3
        s = Solver()
        return str(s.check(And(neq, *symbol_positive))) == 'unsat'

    # enumerate all possible input shardings. -1 means duplicate
    for in_shard_dims in product(*(range(-1, len(i.shape)) for i in inputs)):
        if all((shard_dim == -1 for shard_dim in in_shard_dims)):
            continue

        # Get a tile of each input tensor according to the input sharding
        # And compute the output shape
        try:
            new_inputs = inputs.copy()
            for i, shard_dim in enumerate(in_shard_dims):
                if shard_dim >= 0:
                    new_inputs[i] = _get_first_tile(new_inputs[i], num_shards, shard_dim)
            outputs = op.reforward(new_inputs)
        except (ValueError, RuntimeError, AssertionError):
            continue

        # find the sharded output dimension
        # for each output tensor, there should be at most one dimension being sharded
        out_shard_dims = []
        for i, out in enumerate(outputs):
            shard_dim = -1
            origin_shape = op.outputs[i].shape
            for dim in range(len(out.shape)):
                if _symbol_eq(out.shape[dim], origin_shape[dim] // num_shards):
                    if shard_dim != -1:
                        shard_dim = None  # invalid output shape
                        # Because there should be at most one sharded dimension
                    else:
                        shard_dim = dim
                elif not _symbol_eq(out.shape[dim], origin_shape[dim]):
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
                    found_rules.append(
                        OpShardSpec(in_specs, out_specs, reduce_fn=[ReduceFunction(reduce_op, 'all_reduce')])
                    )
                    # if all_reduce is valid, then reduce_scatter is also valid
                    output = op.task.outputs[0]
                    out_spec = TensorShardSpec(len(output.shape), 0)
                    found_rules.append(
                        OpShardSpec(in_specs, [out_spec], reduce_fn=[ReduceFunction(reduce_op, 'reduce_scatter')])
                    )
    return found_rules


if __name__ == '__main__':
    import hidet.testing

    def resnet_test():
        model = hidet.testing.models.resnet.resnet18()
        x = hidet.symbol(['batch_size', 3, 224, 224])
        flow_graph = hidet.trace_from(model(x))

        cache = {}
        for op in flow_graph.nodes:
            print(op)
            if str(op) not in cache:
                shard_plans = op_shard_rule_search(op, 2)
                cache[str(op)] = shard_plans
            for sp in cache[str(op)]:
                print(sp)

    resnet_test()
