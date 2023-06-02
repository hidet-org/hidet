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
from hashlib import sha256

import hidet
from hidet.ir.type import FuncType, void, byte_p, func_type
from hidet.ir.expr import SymbolVar, Var, Expr, var
from hidet.ir.stmt import AssignStmt, DeclareStmt, BufferStoreStmt
from hidet.graph.tensor import Tensor
from hidet.graph.flow_graph import FlowGraph
from hidet.runtime.compiled_module import CompiledModule
from hidet.runtime.compiled_graph import CompiledGraph, GraphMetaData, GraphExecution, GraphExecutionInstruction
from hidet.runtime.compiled_task import CompiledTask
from hidet.graph.operator import Operator
from hidet.ir import primitives


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
    index_count = 0

    for x in graph.inputs:
        tensor_index[x] = index_count
        index_tensor[index_count] = x
        index_count += 1

    for x in weights:
        tensor_index[x] = index_count
        index_tensor[index_count] = x
        index_count += 1

    graph_execution = GraphExecution()
    graph_execution.weights_index = [tensor_index[x] for x in weights]
    graph_execution.inputs_index = [tensor_index[x] for x in graph.inputs]

    for node_idx, node in enumerate(graph.nodes):

        inst = GraphExecutionInstruction()
        inst.task_idx = node2kernel[node_idx]

        inst.inputs = [tensor_index[x] for x in node.inputs]

        for out in node.outputs:
            tensor_index[out] = index_count
            index_tensor[index_count] = out
            index_count += 1

        inst.outputs = [tensor_index[x] for x in node.outputs]

        for x in node.inputs:
            usage_count[x] -= 1
            if usage_count[x] == 0:
                inst.free.append(tensor_index[x])

        graph_execution.instructions.append(inst)

    graph_execution.outputs_index = [tensor_index[x] for x in graph.outputs]

    return graph_execution


def get_graph_meta_data(graph: FlowGraph, num_kernels) -> GraphMetaData:
    input_signatures = []
    for x in graph.inputs:
        signature = [x.dtype.name]
        for d in x.shape:
            if isinstance(d, int):
                signature.append(d)
            elif isinstance(d, SymbolVar):
                signature.append(d.name)
            else:
                raise RuntimeError(f'Unknown shape type: {d}')
        input_signatures.append(signature)
    output_signatures = []
    for y in graph.outputs:
        signature = [y.dtype.name]
        for d in y.shape:
            if isinstance(d, int):
                signature.append(d)
            else:
                signature.append(str(d))
        output_signatures.append(signature)

    lines = []
    for w in get_graph_weights(graph):
        lines.append(w.signature())
    for node in graph.nodes:
        lines.append(str(node.task))
    lines.append(graph.inputs[0].device.type)
    lines.append(str(graph))
    graph_hash = sha256('\n'.join(lines).encode('utf-8')).hexdigest()[:16]

    return GraphMetaData(
        input_signatures=input_signatures,
        output_signatures=output_signatures,
        device=graph.inputs[0].device.type,
        hidet_version=hidet.__version__,
        num_kernels=num_kernels,
        graph_hash=graph_hash,
    )


def build_graph_module(
    graph: FlowGraph, graph_weights: List[Tensor], node2kernel: List[int], allow_hook=False
) -> CompiledModule:
    from hidet.lang import void_p, attrs, int32, int64, meta, cast
    from hidet.ir.primitives.runtime import memory_planner_init, memory_planner_allocate, memory_planner_free
    from hidet.ir.primitives.runtime import memory_planner_used

    graph_intermediates: List[Tensor] = get_graph_intermediates(graph)
    graph_tensors: List[Tensor] = list(set(graph_weights + graph_intermediates + graph.inputs + graph.outputs))
    tensor_size: Dict[Tensor, Expr] = {x: int64(x.nbytes) for x in graph_tensors}

    graph_nodes: List[Operator] = graph.nodes

    with hidet.script_module() as script_module:

        workspace = script_module.define_global_var('workspace', byte_p)
        weights = script_module.define_global_var('weights', void_p[len(graph_weights)])
        exec_hook = script_module.define_global_var('exec_hook', func_type([~int64], void))

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

        def get_workspace_size_impl(size: Var):
            sb = hidet.ir.builders.StmtBuilder()
            usage_count = graph.usage_count
            tensor_ptr: Dict[Tensor, Var] = {x: var(x.op.name.lower(), int64) for x in graph_intermediates}
            sb += memory_planner_init()
            for node in graph_nodes:
                for y in node.outputs:
                    if y in graph_intermediates:
                        sb += DeclareStmt(tensor_ptr[y], init=memory_planner_allocate(tensor_size[y]))
                sb += AssignStmt(size, primitives.max(size, memory_planner_used()))
                for x in node.inputs:
                    usage_count[x] -= 1
                    if usage_count[x] == 0 and x in graph_intermediates:
                        sb += memory_planner_free(tensor_ptr[x])
            return sb.finish()

        @hidet.script
        def get_workspace_size() -> int64:
            attrs.func_kind = 'public'

            size = int64(0)
            get_workspace_size_impl(size)
            return size

        @hidet.script
        def set_workspace(space: void_p):
            attrs.func_kind = 'public'

            AssignStmt(workspace, space)

        @hidet.script
        def register_hook(hook: void_p):
            attrs.func_kind = 'public'

            assert allow_hook, "Hook is not allowed when building the graph"
            nonlocal exec_hook
            exec_hook = hook

        def call_exec_hook(idx: int, node_params: List[Expr]):
            sb = hidet.ir.builders.StmtBuilder()

            with sb.if_then(exec_hook != 0):
                args = []
                args.append(idx)  # kernel index

                tensors: List[Tensor]
                if idx < len(graph_nodes):
                    args.append(len(graph_nodes[idx].inputs))
                    args.append(len(graph_nodes[idx].outputs))
                    tensors = graph_nodes[idx].inputs + graph_nodes[idx].outputs
                else:
                    args.append(len(graph.inputs))
                    args.append(len(graph.outputs))
                    tensors = graph.inputs + graph.outputs

                assert len(tensors) == len(node_params), "Expect {} parameters, got {}".format(
                    len(tensors), len(node_params)
                )

                for tensor, param in zip(tensors, node_params):
                    args.append(tensor.dtype.name)
                    args.append(len(tensor.shape))
                    args.extend([cast(d, 'int64') for d in tensor.shape])
                    args.append(param)

                args_var = var('args', int64[len(args)])
                sb += DeclareStmt(args_var)
                for i in range(len(args)):
                    sb += BufferStoreStmt(args_var, [i], args[i])
                sb += exec_hook(args_var)
            return sb.finish()

        def launch_impl(inputs: List[Var], outputs: List[Var], p_kernels: Var):
            sb = hidet.ir.builders.StmtBuilder()
            usage_count = graph.usage_count
            tensor_ptr: Dict[Tensor, Var] = {x: var(x.op.name.lower(), int64) for x in graph_intermediates}

            sb += memory_planner_init()
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
                        node_params.append(workspace + tensor_ptr[x])
                    else:
                        raise RuntimeError("Unknown tensor {}".format(x))
                for y in node.outputs:
                    if y in graph_intermediates:
                        sb += DeclareStmt(tensor_ptr[y], init=memory_planner_allocate(tensor_size[y]))
                        node_params.append(workspace + tensor_ptr[y])
                    elif y in graph.outputs:
                        node_params.append(outputs[graph.outputs.index(y)])
                    else:
                        raise RuntimeError("Unknown tensor {}".format(y))

                kernel_type = FuncType([void_p for _ in node_params], void)
                kernel_var = var("k{}_{}".format(idx, graph_nodes[idx].name), kernel_type)
                with sb.let(kernel_var, cast(p_kernels[node2kernel[idx]], kernel_type)):
                    sb += kernel_var(*node_params)
                    if allow_hook:
                        sb += call_exec_hook(idx, node_params)

                for x in node.inputs:
                    usage_count[x] -= 1
                    if usage_count[x] == 0 and x in graph_intermediates:
                        sb += memory_planner_free(tensor_ptr[x])
            if allow_hook:
                sb += call_exec_hook(len(graph_nodes), inputs + outputs)
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


def build_flow_graph(graph, *, space=0, allow_hook=False) -> CompiledGraph:

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
            task_string = str(node.task)
            if task_string not in task2kernel:
                kernel_idx = len(graph_kernels)
                task2kernel[task_string] = kernel_idx
                graph_kernels.append(node.task.build(target=node.build_target))
            node2kernel.append(task2kernel[task_string])

    # build the graph module
    graph_module = build_graph_module(graph, graph_weights, node2kernel, allow_hook=allow_hook)

    # construct the graph execution
    graph_execution = create_graph_execution(graph, graph_weights, node2kernel)

    # get the graph meta data
    graph_meta_data = get_graph_meta_data(graph, len(graph_kernels))

    # build the compiled graph
    compiled_graph = CompiledGraph(
        meta_data=graph_meta_data,
        graph_module=graph_module,
        weights=graph_weights,
        compiled_tasks=graph_kernels,
        graph_execution=graph_execution,
        graph_string=str(graph),
    )

    return compiled_graph
