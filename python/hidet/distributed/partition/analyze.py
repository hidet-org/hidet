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

from typing import Dict, List, Set, Tuple
from itertools import product
import tqdm

from mip import Model, BINARY, xsum, minimize, OptimizationStatus

import hidet
from hidet.graph import FlowGraph, Operator, Tensor

from .rule import op_shard_rule_search
from .shard import OpShardSpec, TensorShardSpec, connect, node_comm_cost

# I copied it from compiled_graph.py
def get_graph_weights(graph):
    """
    Get the weights of the graph. All constant tensors used by the operators in the graph, or returned directly by the
    graph, are considered as weights.
    """
    weights: Set[Tensor] = set()
    for node in graph.nodes:
        for x in node.inputs:
            if x.storage is not None:
                weights.add(x)
    for y in graph.outputs:
        if y.storage is not None:
            weights.add(y)
    return list(weights)


def generate_rules(g: FlowGraph, num_shards: int) -> Dict[Operator, List[OpShardSpec]]:
    cache = {}
    ret = {}
    for node in tqdm.tqdm(g.nodes):
        node_str = str(node)
        if node_str not in cache:
            cache[node_str] = op_shard_rule_search(node, num_shards)
        ret[node] = cache[node_str]
    return ret


def search_strategy(
    g: FlowGraph, num_shards: int, mem_budget: int, verbose=0, max_seconds=None
) -> Tuple[Dict[Operator, OpShardSpec], Dict[Tensor, TensorShardSpec]]:
    # Only suport 1D partition
    m = Model()
    m.verbose = verbose
    m.threads = -1
    print("Generating rules for each op...")
    op_rules = generate_rules(g, num_shards)

    print("Building ILP...")
    parameters = get_graph_weights(g)
    param_vars = {}
    param_mem = 0
    param_tot = 0
    for p in parameters:
        p_vars = [m.add_var(var_type=BINARY) for _ in p.shape]
        m += xsum(p_vars) <= 1
        param_vars[p] = p_vars
        sharded = xsum(p_vars)
        param_mem += (num_shards - ((num_shards - 1) * sharded)) * (p.nbytes // num_shards)
        param_tot += p.nbytes
    print(f"Total paramter size: {param_tot/1024**3} GiB")
    m += param_mem <= mem_budget

    # node communication cost
    node_vars = {}
    node_cost = 0
    for node in g.nodes:
        assert len(op_rules[node]) > 0
        rule_cost = [node_comm_cost(node.outputs, rule) for rule in op_rules[node]]
        node_vars[node] = [m.add_var(var_type=BINARY) for _ in op_rules[node]]
        node_cost += xsum(c * v for c, v in zip(rule_cost, node_vars[node]))
        for rid, rule in enumerate(op_rules[node]):
            for in_tensor, spec in zip(node.inputs, rule.input_specs):
                if in_tensor in parameters:
                    rule_var = node_vars[node][rid]
                    p_vars = param_vars[in_tensor]
                    if spec.is_full():
                        m += xsum(p_vars) + rule_var <= 1  # If full parameter is required, do not shard
                    else:
                        shard_dim = spec.sharded_dim()
                        for i in range(len(in_tensor.shape)):
                            if i == shard_dim:
                                m += p_vars[shard_dim] >= rule_var
                            else:
                                m += rule_var + p_vars[i] <= 1

        m += xsum(node_vars[node]) == 1

    # cross node communication cost
    edge_cost = 0
    for v in g.nodes:
        # allow multiple edges. one edge represents one tensor.
        # u -> v
        for input_idx, input_tensor in enumerate(v.inputs):
            if input_tensor.op is not None:
                u = input_tensor.op
                u_rules = op_rules[u]
                v_rules = op_rules[v]

                uv_vars = []
                for i, u_rule in enumerate(u_rules):
                    for j, v_rule in enumerate(v_rules):
                        var = m.add_var(var_type=BINARY)
                        uv_vars.append(var)
                        m += var <= node_vars[u][i]
                        m += var <= node_vars[v][j]
                        m += var >= node_vars[u][i] + node_vars[v][j] - 1

                        u_spec = u_rule.output_specs[u.outputs.index(input_tensor)]
                        v_spec = v_rule.input_specs[input_idx]
                        edge_cost += var * connect(input_tensor, u_spec, v_spec)[1]
                m += xsum(uv_vars) == 1
    

    m.objective = minimize(node_cost + edge_cost)
    print("Solving...")
    status = m.optimize(max_seconds=max_seconds)
    assert status in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)
    print(f"Minimal cost: {m.objective.x}")
    print(f"Paramater Size: {param_mem.x / 1024**3} GiB")

    op_specs = {}
    for node in g.nodes:
        for i, v in enumerate(node_vars[node]):
            if v.x >= 0.99:
                op_specs[node] = op_rules[node][i]
        assert node in op_specs, str(node) + str([v.x for v in node_vars[node]])

    param_specs = {}
    for param in parameters:
        for i, v in enumerate(param_vars[param]):
            if v.x >= 0.99:
                param_specs[param] = TensorShardSpec(len(param.shape), i)
                break
        if param not in param_specs:
            param_specs[param] = TensorShardSpec(len(param.shape))

    return op_specs, param_specs
