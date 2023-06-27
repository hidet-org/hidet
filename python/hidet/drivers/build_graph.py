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
from typing import List, Set, Dict
import os
import json
import shutil
from hashlib import sha256

import hidet
from hidet.ir.type import FuncType, void, byte_p
from hidet.ir.expr import SymbolVar, Var, Expr, var
from hidet.ir.stmt import AssignStmt, DeclareStmt
from hidet.graph.tensor import Tensor
from hidet.graph.flow_graph import FlowGraph
from hidet.runtime.compiled_module import CompiledModule
from hidet.runtime.compiled_graph import (
    CompiledGraph,
    GraphMetaData,
    GraphExecution,
    GraphExecutionInstruction,
)
from hidet.runtime.compiled_task import CompiledTask, TensorSignature
from hidet.graph.operator import Operator
from hidet.ir import primitives
from hidet.utils.dataclass import asdict


def get_graph_weights(graph):
    weights: Set[Tensor] = set()
    for node in graph.nodes:
        for x in node.inputs:
            if x.storage is not None:
                weights.add(x)
    return list(weights)


def get_graph_intermediates(graph):
    intermediates: List[Tensor] = []
    for node in graph.nodes:
        for y in node.outputs:
            if y in graph.outputs:
                continue
            intermediates.append(y)
    return intermediates


def create_graph_execution(graph: FlowGraph, weights: List[Tensor], node2kernel: List[int]) -> GraphExecution:
    usage_count: Dict[Tensor, int] = graph.usage_count

    tensor_index: Dict[Tensor, int] = {}
    index_tensor: Dict[int, Tensor] = {}
    tensor_device: List[str] = []
    index_count = 0

    for x in graph.inputs:
        tensor_index[x] = index_count
        index_tensor[index_count] = x
        tensor_device.append(x.device.kind)
        index_count += 1

    for x in weights:
        tensor_index[x] = index_count
        index_tensor[index_count] = x
        tensor_device.append(x.device.kind)
        index_count += 1

    weights_index = [tensor_index[x] for x in weights]
    inputs_index = [tensor_index[x] for x in graph.inputs]
    instructions: List[GraphExecutionInstruction] = []

    for node_idx, node in enumerate(graph.nodes):
        inst_task_idx = node2kernel[node_idx]

        inst_inputs = [tensor_index[x] for x in node.inputs]

        for out in node.outputs:
            tensor_index[out] = index_count
            index_tensor[index_count] = out
            tensor_device.append(out.device.kind)
            index_count += 1

        inst_outputs = [tensor_index[x] for x in node.outputs]
        inst_free = []
        for x in node.inputs:
            usage_count[x] -= 1
            if usage_count[x] == 0:
                inst_free.append(tensor_index[x])

        instructions.append(GraphExecutionInstruction(inst_task_idx, inst_inputs, inst_outputs, inst_free))

    outputs_index = [tensor_index[x] for x in graph.outputs]

    return GraphExecution(
        weights_index=weights_index,
        inputs_index=inputs_index,
        outputs_index=outputs_index,
        instructions=instructions,
        tensor_device=tensor_device,
    )


def get_graph_meta_data(graph: FlowGraph, num_kernels, space: int) -> GraphMetaData:
    # input tensor signature
    inputs = []
    for x in graph.inputs:
        shape = []
        for d in x.shape:
            if isinstance(d, int):
                shape.append(d)
            elif isinstance(d, SymbolVar):
                shape.append(d.name)
            else:
                raise RuntimeError('Graph input shape must be either int or symbolic var, but got {}'.format(d))
        inputs.append(TensorSignature(device=x.device.kind, dtype=x.dtype.name, shape=shape))

    # output tensor signature
    outputs = []
    for y in graph.outputs:
        shape = [int(d) if isinstance(d, int) else str(d) for d in y.shape]
        outputs.append(TensorSignature(device=y.device.kind, dtype=y.dtype.name, shape=shape))

    # graph hash
    lines = []
    for x in graph.inputs:
        lines.append(x.signature())
    for y in graph.outputs:
        lines.append(y.signature())
    for w in get_graph_weights(graph):
        lines.append(w.signature())
    for node in graph.nodes:
        lines.append(str(node.task))
    lines.append(str(graph))
    lines.append(str(space))
    graph_hash = sha256('\n'.join(lines).encode('utf-8')).hexdigest()[:16]

    return GraphMetaData(
        inputs=inputs, outputs=outputs, hidet_version=hidet.__version__, num_kernels=num_kernels, graph_hash=graph_hash
    )

def build_graph_module(graph: FlowGraph, graph_weights: List[Tensor], node2kernel: List[int]) -> CompiledModule:
    from hidet.lang import void_p, attrs, int32, int64, meta, cast
    from hidet.ir.primitives.runtime import memory_planner_init, memory_planner_allocate, memory_planner_free
    from hidet.ir.primitives.runtime import memory_planner_used

    graph_intermediates: List[Tensor] = get_graph_intermediates(graph)
    graph_tensors: List[Tensor] = list(set(graph_weights + graph_intermediates + graph.inputs + graph.outputs))
    tensor_size: Dict[Tensor, Expr] = {x: int64(x.nbytes) for x in graph_tensors}

    graph_nodes: List[Operator] = graph.nodes

    with hidet.script_module() as script_module:
        cpu_workspace = script_module.define_global_var('cpu_workspace', byte_p)
        cuda_workspace = script_module.define_global_var('cuda_workspace', byte_p)
        weights = script_module.define_global_var('weights', void_p[len(graph_weights)])

        @hidet.script
        def init(num_weights: int, p_weights: ~void_p):
            attrs.func_kind = 'public'
            assert num_weights == len(graph_weights), "Expect {} weights".format(len(graph_weights))
            for i in range(len(graph_weights)):
                weights[i] = p_weights[i]

        @hidet.script
        def get_output_shape(index: int, dims: ~int32):
            attrs.func_kind = 'public'

            for idx in meta.each(range(len(graph.outputs))):
                if idx == index:
                    for dim_idx, dim in meta.each(enumerate(graph.outputs[idx].shape)):
                        dims[dim_idx] = dim

        def get_workspace_size_impl(cpu_size: Var, cuda_size: Var):
            sb = hidet.ir.builders.StmtBuilder()
            usage_count = graph.usage_count
            tensor_ptr: Dict[Tensor, Var] = {x: var(x.op.name.lower(), int64) for x in graph_intermediates}
            cpu_idx = 0  # memory planner index
            cuda_idx = 1
            device2idx = {'cpu': cpu_idx, 'cuda': cuda_idx}
            for idx in [cpu_idx, cuda_idx]:
                sb += memory_planner_init(idx)
            for node in graph_nodes:
                for y in node.outputs:
                    if y in graph_intermediates:
                        sb += DeclareStmt(
                            tensor_ptr[y], init=memory_planner_allocate(device2idx[y.device.kind], tensor_size[y])
                        )
                sb += AssignStmt(cpu_size, primitives.max(cpu_size, memory_planner_used(cpu_idx)))
                sb += AssignStmt(cuda_size, primitives.max(cuda_size, memory_planner_used(cuda_idx)))
                for x in node.inputs:
                    usage_count[x] -= 1
                    if usage_count[x] == 0 and x in graph_intermediates:
                        sb += memory_planner_free(device2idx[x.device.kind], tensor_ptr[x])
            return sb.finish()

        @hidet.script
        def get_workspace_size(sizes: int64[2]):
            attrs.func_kind = 'public'

            cpu_size = int64(0)
            cuda_size = int64(0)
            get_workspace_size_impl(cpu_size, cuda_size)
            sizes[0] = cpu_size
            sizes[1] = cuda_size

        @hidet.script
        def set_workspace(idx: int32, space: void_p):
            attrs.func_kind = 'public'

            assert 0 <= idx < 2, "Invalid workspace index"

            if idx == 0:
                AssignStmt(cpu_workspace, space)
            else:
                AssignStmt(cuda_workspace, space)

        def launch_impl(inputs: List[Var], outputs: List[Var], p_kernels: Var):
            sb = hidet.ir.builders.StmtBuilder()
            usage_count = graph.usage_count
            tensor_ptr: Dict[Tensor, Var] = {x: var(x.op.name.lower(), int64) for x in graph_intermediates}

            sb += memory_planner_init(0)
            sb += memory_planner_init(1)
            d2i = {'cpu': 0, 'cuda': 1}
            d2w = {'cpu': cpu_workspace, 'cuda': cuda_workspace}
            for idx, node in enumerate(graph_nodes):
                node_params = []
                for x in node.inputs:
                    if x in graph.inputs:
                        node_params.append(inputs[graph.inputs.index(x)])
                    elif x in graph_weights:
                        node_params.append(weights[graph_weights.index(x)])
                    elif x in graph.outputs:
                        node_params.append(outputs[graph.outputs.index(x)])
                    elif x in graph_intermediates:
                        node_params.append(d2w[x.device.kind] + tensor_ptr[x])
                    else:
                        raise RuntimeError("Unknown tensor {}".format(x))
                for y in node.outputs:
                    if y in graph_intermediates:
                        sb += DeclareStmt(
                            tensor_ptr[y], init=memory_planner_allocate(d2i[y.device.kind], tensor_size[y])
                        )
                        node_params.append(d2w[y.device.kind] + tensor_ptr[y])
                    elif y in graph.outputs:
                        node_params.append(outputs[graph.outputs.index(y)])
                    else:
                        raise RuntimeError("Unknown tensor {}".format(y))

                kernel_type = FuncType([void_p for _ in node_params], void)
                kernel_var = var("k{}_{}".format(idx, graph_nodes[idx].name), kernel_type)
                with sb.let(kernel_var, cast(p_kernels[node2kernel[idx]], kernel_type)):
                    sb += kernel_var(*node_params)

                for x in node.inputs:
                    usage_count[x] -= 1
                    if usage_count[x] == 0 and x in graph_intermediates:
                        sb += memory_planner_free(d2i[x.device.kind], tensor_ptr[x])
            return sb.finish()

        @hidet.script
        def launch(
            inputs: meta.types([void_p for _ in graph.inputs]),
            outputs: meta.types([void_p for _ in graph.outputs]),
            p_kernels: ~void_p,
        ):
            attrs.func_kind = 'public'

            launch_impl(inputs, outputs, p_kernels)

    return script_module.build()


def save_to_graph_cache(cgraph: CompiledGraph):
    cache_dir = hidet.utils.cache_dir('graphs', cgraph.meta.graph_hash)

    # save meta data
    with open(os.path.join(cache_dir, 'meta.json'), 'w') as f:
        json.dump(asdict(cgraph.meta), f, indent=4)

    # save graph module
    shutil.copytree(cgraph.graph_module.module_dir, os.path.join(cache_dir, 'graph_module/'), dirs_exist_ok=True)

    # save kernels
    for i, compiled_task in enumerate(cgraph.compiled_tasks):
        shutil.copytree(compiled_task.task_dir, os.path.join(cache_dir, 'kernels/{}'.format(i)), dirs_exist_ok=True)

    # save graph execution
    with open(os.path.join(cache_dir, 'graph_execution.json'), 'w') as f:
        json.dump(asdict(cgraph.graph_execution), f, indent=4)

    # save graph string
    with open(os.path.join(cache_dir, 'graph_string.txt'), 'w') as f:
        f.write(cgraph.graph_string)


def build_flow_graph(graph, *, space=0) -> CompiledGraph:
    assert isinstance(graph, FlowGraph)

    # get the graph weights
    graph_weights: List[Tensor] = get_graph_weights(graph)

    # get the graph kernels
    with hidet.option.context():
        hidet.option.search_space(space)

        graph._build_nodes()  # pylint: disable=protected-access
        graph_kernels: List[CompiledTask] = []
        task2kernel: Dict[str, int] = {}
        node2kernel: List[int] = []
        for node in graph.nodes:
            task_string = str(node.task) + ' space: {}'.format(space) + ' target: {}'.format(node.build_target)
            if task_string not in task2kernel:
                kernel_idx = len(graph_kernels)
                task2kernel[task_string] = kernel_idx
                graph_kernels.append(node.task.build(target=node.build_target))
            node2kernel.append(task2kernel[task_string])

    # build the graph module
    graph_module = build_graph_module(graph, graph_weights, node2kernel)

    # construct the graph execution
    graph_execution = create_graph_execution(graph, graph_weights, node2kernel)

    # get the graph meta data
    graph_meta_data = get_graph_meta_data(graph, len(graph_kernels), space)

    # build the compiled graph
    compiled_graph = CompiledGraph(
        meta=graph_meta_data,
        graph_module=graph_module,
        weights=graph_weights,
        compiled_tasks=graph_kernels,
        graph_execution=graph_execution,
        graph_string=str(graph),
    )

    # save the compiled graph to cache
    save_to_graph_cache(compiled_graph)

    return compiled_graph
