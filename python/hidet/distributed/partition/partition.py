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
import os
import pickle

import hidet
from hidet.graph import FlowGraph, Operator, Tensor
from hidet.graph.graph_utils.functors import GraphCloneRewriter

from .analyze import search_strategy
from .shard import OpShardSpec, TensorShardSpec, connect
from .rule import get_tile


class CommunicationInjection(GraphCloneRewriter):
    def __init__(
        self,
        g: FlowGraph,
        num_shards: int,
        rank: int,
        op_specs: Dict[Operator, OpShardSpec],
        param_specs: Dict[Tensor, TensorShardSpec],
    ):
        super().__init__()
        self.g = g
        self.num_shards = num_shards
        self.op_specs = op_specs
        self.param_specs = param_specs
        self.rank = rank

    def _tensor_spec(self, tensor: Tensor):
        if tensor.op is not None:
            op = tensor.op
            return self.op_specs[op].output_specs[op.outputs.index(tensor)]
        elif not tensor.is_symbolic(): # parameter
            return self.param_specs[tensor]
        else:  # input. We assume input is not sharded
            assert tensor in self.g.inputs
            return TensorShardSpec(len(tensor.shape))

    def visit_FlowGraph(self, graph: FlowGraph):
        outputs = [self.visit(output) for output in graph.outputs]

        # Check outputs and gather if they're sharded
        for i, out in enumerate(outputs):
            target_spec = TensorShardSpec(len(out.shape))
            convert, _ = connect(out, self._tensor_spec(self.g.outputs[i]), target_spec)
            if convert is not None:
                outputs[i] = convert(out, self.num_shards, self.rank)
        return FlowGraph(outputs, graph.inputs)

    def visit_Operator(self, op: Operator):
        inputs_from_pred = [self(x) for x in op.inputs]
        given_specs = [self._tensor_spec(x) for x in op.inputs]
        op_spec = self.op_specs[op] 
        required_specs = op_spec.input_specs

        # align pred_specs and in_specs
        updated_inputs = []
        for full_input, required_spec, updated_input, given_spec in zip(
            op.inputs, required_specs, inputs_from_pred, given_specs
        ):
            convert, _ = connect(full_input, given_spec, required_spec)
            if convert is not None:
                updated_input = convert(updated_input, self.num_shards, self.rank)
            updated_inputs.append(updated_input)

        updated_outputs = op.reforward(updated_inputs)
        reduced_outputs = []

        for out, reduce_fn in zip(updated_outputs, op_spec.reduce_fn):
            if reduce_fn is not None:
                out = reduce_fn(out, self.num_shards)
            reduced_outputs.append(out)

        for original, updated in zip(op.outputs, reduced_outputs):
            self.memo[original] = updated

    def visit_Tensor(self, tensor: Tensor):
        if tensor.trace is None:
            # keep the input tensor the same
            if tensor in self.param_specs:
                spec = self.param_specs[tensor]
                if not spec.is_full():
                    shard_dim = spec.sharded_dim()
                    assert tensor.shape[shard_dim] % self.num_shards == 0
                    self.memo[tensor] = get_tile(tensor, self.num_shards, shard_dim, self.rank)
                    return self.memo[tensor]
            return tensor
        else:
            self(tensor.trace[0])
            return self.memo[tensor]


def partition(g: FlowGraph, distributed_config: dict, out_dir: str) -> None:
    n_shards = distributed_config['ngpus']
    mem_budget = distributed_config['mem_budget']
    max_seconds = distributed_config.get('search_max_seconds', 0)
    solver_verbose = distributed_config.get('solver_verbose', 0)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"origin_graph.txt"), 'w') as f:
        print(g, file=f)

    op_specs, param_specs = search_strategy(g, n_shards, mem_budget, max_seconds=max_seconds, verbose=solver_verbose)
    pickle.dump((op_specs, param_specs), open(os.path.join(out_dir, "op_param_shard.data"), "wb"))
    parts, op_specs, param_specs = _partition(g, n_shards, op_specs, param_specs)
    for rank, part in enumerate(parts):
        part.save(os.path.join(out_dir, f"part{rank}.graph"))

        # for debug purpose
        with open(os.path.join(out_dir, f"part{rank}.txt"), 'w') as f:
            print(part, file=f)

    with open(os.path.join(out_dir, f"scheme.txt"), 'w') as f:
        for node in g.nodes:
            print(node, op_specs[node], file=f)


def _partition(
    g: FlowGraph, num_shards: int, op_specs: Dict[Operator, OpShardSpec], param_specs: Dict[Tensor, TensorShardSpec]
) -> List[FlowGraph]:
    # There should be a smart way to inject rank
    parts = []
    for rank in range(num_shards):
        print(f"Generating part {rank}")
        comm_inject = CommunicationInjection(g, num_shards, rank, op_specs, param_specs)
        part = comm_inject(g)
        parts.append(part)
    return parts, op_specs, param_specs # for debugging
