from typing import List, Set, Dict

import hidet
from hidet.ir.type import FuncType, void, byte_p
from hidet.ir.expr import SymbolVar, Var, Expr, var
from hidet.ir.stmt import AssignStmt, DeclareStmt
from hidet.graph.tensor import Tensor
from hidet.graph.flow_graph import FlowGraph
from hidet.runtime.module import CompiledModule
from hidet.runtime.model import CompiledModel, ModelMetaData
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


def get_graph_meta_data(graph: FlowGraph) -> ModelMetaData:
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
    return ModelMetaData(
        input_signatures=input_signatures,
        output_signatures=output_signatures,
        device=graph.inputs[0].device.type,
        hidet_version=hidet.__version__,
        num_kernels=len(graph.nodes),
    )


def flow_graph_build(graph) -> CompiledModel:
    from hidet.lang import void_p, attrs, int32, int64, meta, cast
    from hidet.ir.primitives.runtime import memory_planner_init, memory_planner_allocate, memory_planner_free
    from hidet.ir.primitives.runtime import memory_planner_used

    assert isinstance(graph, FlowGraph)

    graph_weights: List[Tensor] = get_graph_weights(graph)
    graph_intermediates: List[Tensor] = get_graph_intermediates(graph)
    graph_tensors: List[Tensor] = list(set(graph_weights + graph_intermediates + graph.inputs + graph.outputs))
    tensor_size: Dict[Tensor, Expr] = {x: int64(x.nbytes) for x in graph_tensors}
    graph_nodes: List[Operator] = graph.nodes

    with hidet.script_module() as script_module:

        workspace = var('workspace', byte_p)
        weights = var('weights', void_p[len(graph_weights)])
        kernels = var('kernels', void_p[len(graph_nodes)])

        script_module.define_global_var(workspace)
        script_module.define_global_var(weights)
        script_module.define_global_var(kernels)

        @hidet.script
        def init(num_kernels: int, p_kernels: ~void_p, num_weights: int, p_weights: ~void_p):
            attrs.func_kind = 'public'
            assert num_kernels == len(graph_nodes), "Expect {} kernels".format(len(graph_nodes))
            assert num_weights == len(graph_weights), "Expect {} weights".format(len(graph_weights))
            for i in range(len(graph_nodes)):
                kernels[i] = p_kernels[i]
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

        def launch_impl(inputs: List[Var], outputs: List[Var]):
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

                func_type = FuncType([void_p for _ in node_params], void)
                kernel_var = var("kernel_{}".format(idx), func_type)
                with sb.let(kernel_var, cast(kernels[idx], func_type)):
                    sb += kernel_var(*node_params)

                for x in node.inputs:
                    usage_count[x] -= 1
                    if usage_count[x] == 0 and x in graph_intermediates:
                        sb += memory_planner_free(tensor_ptr[x])
            return sb.finish()

        @hidet.script
        def launch(
            inputs: meta.types([void_p for _ in graph.inputs]), outputs: meta.types([void_p for _ in graph.outputs])
        ):
            attrs.func_kind = 'public'

            launch_impl(inputs, outputs)

    graph_module = script_module.build()

    graph._build_nodes()  # pylint: disable=protected-access
    graph_kernels: List[CompiledModule] = [node.task_func for node in graph.nodes]

    graph_meta_data = get_graph_meta_data(graph)
    compiled_model = CompiledModel(graph_meta_data, graph_module, graph_weights, graph_kernels)

    return compiled_model
