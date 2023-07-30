from typing import List
from itertools import product
from collections import OrderedDict

from z3 import Int, And, Or, Not, Solver, Const, IntSort

from .shard import TensorShardSpec, OpShardSpec

import hidet
from hidet.graph import Operator
from hidet.ir import TensorElement, TensorNode
from hidet.ir.tools import collect
from hidet.ir.compute import GridCompute, ReduceCompute, TensorInput
from hidet.ir.functors import ExprVisitor, ComputeVisitor


class IndexExprBuilder(ExprVisitor):
    def process_expr(self, e, axis_z3):
        self.axis_z3 = axis_z3
        return self.visit(e)
    
    def visit_Var(self, e):
        assert e in self.axis_z3
        return self.axis_z3[e]
    
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
        return Const(val, IntSort())
    
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

class DataDependencyAnalyzer:
    def __init__(self, op: Operator, num_shards: int):
        self.op = op
        self.num_shards = num_shards
        self.valid = self._build_bound_expr()

    def _build_bound_expr(self) -> bool:
        task = self.op.task
        outputs = task.outputs
        index_builder = IndexExprBuilder()

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
                return False# we don't support scalar output
            axis_map = {}
            for axis in o.axes:
                axis_map[axis] = Int(f'i_{len(self.grid_z3_vars)}')
                self.grid_z3_vars[axis] = axis_map[axis]
            axis_map.update(self.reduce_z3_vars)
            _build_grid_bound(o, axis_map)
        return True

    def check(self, in_shard_dim: List[int], out_shard_dim: List[int]):
        # Now only supports 1D partition
        task = self.op.task
        rank_z3 = Int('rank')
        rank_constraint = And(rank_z3 >= 0, rank_z3 < self.num_shards)
        out_and_reduce_constraints = []
        for o, shard_dim in zip(task.outputs, out_shard_dim):
            assert isinstance(o, GridCompute)
            for i, (axis, shape) in enumerate(zip(o.axes, o.shape)):
                axis_z3 = self.grid_z3_vars[axis]
                if i == shard_dim:
                    shard_size = shape // self.num_shards
                    c = And(axis_z3 >= rank_z3 * shard_size, axis_z3 < (rank_z3 + 1) * shard_size)
                else:
                    c = And(axis_z3 >= 0, axis_z3 < shape)
                out_and_reduce_constraints.append(c)
            
            for axis, axis_z3 in self.reduce_z3_vars.items():
                shape = self.reduce_axis_shapes[axis]
                c = And(axis_z3 >= 0, axis_z3 < shape)
                out_and_reduce_constraints.append(c)
        
        in_constraints = []
        for inp, shard_dim in zip(task.inputs, in_shard_dim):
            if inp not in self.input_boundaries:
                continue
            for indices in self.input_boundaries[inp]:
                for i, (axis_z3, shape) in enumerate(zip(indices, inp.shape)):
                    if i == shard_dim:
                        shard_size = shape // self.num_shards
                        c = And(axis_z3 >= rank_z3 * shard_size, axis_z3 < (rank_z3 + 1) * shard_size)
                    else:
                        c = And(axis_z3 >= 0, axis_z3 < shape)
                    in_constraints.append(Not(c))
        
        constraints = [rank_constraint, Or(*in_constraints)] + out_and_reduce_constraints
        s = Solver()
        s.check(*constraints)
        return len(s.model()) == 0

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
    # enumerate all possible input shardings. -1 means duplicate
    data_dependency_analyzer = DataDependencyAnalyzer(op, num_shards)
    if not data_dependency_analyzer.valid:
        return []
    for in_shard_dims in product(*(range(-1, len(i.shape)) for i in inputs)):
        if all((shard_dim == -1 for shard_dim in in_shard_dims)):
            continue
        
        # compute the output shape
        try:
            new_inputs = inputs.copy()
            for i, shard_dim in enumerate(in_shard_dims):
                if shard_dim >= 0:
                    new_inputs[i] = _get_first_tile(new_inputs[i], num_shards, shard_dim)
            outputs = op.reforward(new_inputs)
        except Exception:
            continue

        # find the sharded output dimension
        # for each output tensor, there should be at most one dimension being sharded
        out_shard_dims = []
        for i, out in enumerate(outputs):
            shard_dim = -1
            origin_shape = op.outputs[i].shape
            for dim in range(len(out.shape)):
                if out.shape[dim] == op.outputs[i].shape[dim] // num_shards:
                    if shard_dim != -1:
                        shard_dim = None # invalid output shape
                    else:
                        shard_dim = dim
                elif origin_shape[dim] != out.shape[dim]:
                    shard_dim = None # invalid output shape
            if shard_dim is not None: # output is invalid result of a sharding
                out_shard_dims.append(shard_dim)
            else:
                break
        
        if len(out_shard_dims) == len(outputs): # shape analysis valid
            valid = data_dependency_analyzer.check(in_shard_dims, out_shard_dims)
            if valid:
                in_specs = [TensorShardSpec(len(i.shape), shard_dim) for i, shard_dim in zip(inputs, in_shard_dims)]
                out_specs = [TensorShardSpec(len(o.shape), shard_dim) for o, shard_dim in zip(outputs, out_shard_dims)]
                found_rules.append(OpShardSpec(in_specs, out_specs))
            # data dependency analysis
    for r in found_rules:
        print(r)
    return found_rules



if __name__ == '__main__':
    import hidet.testing
    model = hidet.testing.models.resnet.resnet18()
    x = hidet.symbol([8, 3, 224, 224])
    flow_graph = hidet.trace_from(model(x))

    cache = {}
    for op in flow_graph.nodes:
        print(op)
        if str(op) not in cache:
            shard_plans = op_shard_rule_search(op, 2)
            cache[str(op)] = shard_plans
        for sp in cache[str(op)]:
            print(sp)