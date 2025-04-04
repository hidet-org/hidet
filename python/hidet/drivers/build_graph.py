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
import json
from hashlib import sha256
import numpy
import hidet
from hidet.ir.type import FuncType, void, byte_p
from hidet.ir.expr import SymbolVar, Var, Expr, var
from hidet.ir.stmt import AssignStmt, DeclareStmt
from hidet.graph.tensor import Tensor
from hidet.graph.flow_graph import FlowGraph
from hidet.runtime.compiled_module import CompiledModule
from hidet.runtime.compiled_graph import CompiledGraph, GraphMetaData, GraphExecution, GraphExecutionInstruction
from hidet.runtime.compiled_task import CompiledTask, TensorSignature
from hidet.graph.operator import Operator
from hidet.ir import primitives
from hidet.utils.dataclass import asdict
from hidet.utils import copy_tree_ignore_existing


def get_graph_weights(graph):
    """
    Get the weights of the graph. All constant tensors used by the operators in the graph, or returned directly by the
    graph, are considered as weights.
    """
    weights: List[Tensor] = []
    for node in graph.nodes:
        for x in node.inputs:
            if x.storage is not None:
                weights.append(x)
    for y in graph.outputs:
        if y.storage is not None:
            weights.append(y)
    return weights


def get_graph_intermediates(graph):
    """
    Get the intermediate tensors of the graph: {output tensors of nodes} - {output tensors of the graph}
    """
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

    def add_index_for_tensor(x):
        nonlocal index_count
        if x in tensor_index:
            return
        tensor_index[x] = index_count
        index_tensor[index_count] = x
        tensor_device.append(x.device.kind)
        index_count += 1

    for x in graph.inputs:
        add_index_for_tensor(x)

    for w in weights:
        add_index_for_tensor(w)

    weights_index = [tensor_index[x] for x in weights]
    inputs_index = [tensor_index[x] for x in graph.inputs]
    instructions: List[GraphExecutionInstruction] = []

    for node_idx, node in enumerate(graph.nodes):
        inst_task_idx = node2kernel[node_idx]

        inst_inputs = [tensor_index[x] for x in node.inputs]

        for out in node.outputs:
            add_index_for_tensor(out)

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
        inputs=inputs,
        outputs=outputs,
        hidet_version=hidet.__version__,
        num_kernels=num_kernels,
        graph_hash=graph_hash,
        share_map=graph.share_map,
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
        hip_workspace = script_module.define_global_var('hip_workspace', byte_p)
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

        def get_workspace_size_impl(cpu_size: Var, cuda_size: Var, hip_size: Var):
            sb = hidet.ir.builders.StmtBuilder()
            usage_count = graph.usage_count
            tensor_ptr: Dict[Tensor, Var] = {x: var(x.op.name.lower(), int64) for x in graph_intermediates}
            cpu_idx = 0  # memory planner index
            cuda_idx = 1
            hip_idx = 2
            device2idx = {'cpu': cpu_idx, 'cuda': cuda_idx, 'hip': hip_idx}
            for idx in [cpu_idx, cuda_idx, hip_idx]:
                sb += memory_planner_init(idx)
            for node in graph_nodes:
                for output_idx, y in enumerate(node.outputs):
                    if y in graph_intermediates:
                        if node.share_map and y in node.share_map:
                            # share the memory with input tensor
                            input_idx: int = node.share_map[output_idx]
                            init_addr = tensor_ptr[node.inputs[input_idx]]
                        else:
                            init_addr = memory_planner_allocate(device2idx[y.device.kind], tensor_size[y])
                        sb += DeclareStmt(tensor_ptr[y], init=init_addr)
                sb += AssignStmt(cpu_size, primitives.max(cpu_size, memory_planner_used(cpu_idx)))
                sb += AssignStmt(cuda_size, primitives.max(cuda_size, memory_planner_used(cuda_idx)))
                sb += AssignStmt(hip_size, primitives.max(hip_size, memory_planner_used(hip_idx)))
                for x in node.inputs:
                    usage_count[x] -= 1
                    if usage_count[x] == 0 and x in graph_intermediates:
                        sb += memory_planner_free(device2idx[x.device.kind], tensor_ptr[x])
            return sb.finish()

        @hidet.script
        def get_workspace_size(sizes: int64[3]):
            attrs.func_kind = 'public'

            cpu_size = int64(0)
            cuda_size = int64(0)
            hip_size = int64(0)
            get_workspace_size_impl(cpu_size, cuda_size, hip_size)
            sizes[0] = cpu_size
            sizes[1] = cuda_size
            sizes[2] = hip_size

        @hidet.script
        def set_workspace(idx: int32, space: void_p):
            attrs.func_kind = 'public'

            assert 0 <= idx < 3, "Invalid workspace index"

            if idx == 0:
                AssignStmt(cpu_workspace, space)
            elif idx == 1:
                AssignStmt(cuda_workspace, space)
            else:
                AssignStmt(hip_workspace, space)

        def launch_impl(inputs: List[Var], outputs: List[Var], p_kernels: Var):
            intermediate_vars = [var(x.op.name.lower(), int64) for x in graph_intermediates]
            # Here we store all correspondence between tensors and variables
            # that store address allocated for these Tensors
            t_mapping = Tensor2VarMap(
                graph.inputs,
                inputs,
                graph.outputs,
                outputs,
                graph_weights,
                weights,
                graph_intermediates,
                intermediate_vars,
                graph.usage_count,
                cpu_workspace,
                cuda_workspace,
            )

            sb = hidet.ir.builders.StmtBuilder()
            sb += memory_planner_init(0)
            sb += memory_planner_init(1)
            d2i = {'cpu': 0, 'cuda': 1, 'hip': 2}

            # Apply share_map optimization
            t_mapping.process_share_map(graph_nodes)

            # For every node in graph generate a call of kernel
            for idx, node in enumerate(graph_nodes):
                # 1. Prepare input and output arguments
                node_params = []
                for x in node.inputs:
                    node_params.append(t_mapping.get_full_addr(x))
                for y in node.outputs:
                    if not t_mapping.is_allocated(y) and t_mapping.is_local(y):
                        init_addr = memory_planner_allocate(d2i[y.device.kind], tensor_size[y])
                        sb += DeclareStmt(t_mapping.get_var(y), init=init_addr)
                        t_mapping.set_allocated(y, True)
                    node_params.append(t_mapping.get_full_addr(y))
                # 2. Call a kernel
                kernel_type = FuncType([void_p for _ in node_params], void)
                kernel_var = var("k{}_{}".format(idx, graph_nodes[idx].name), kernel_type)
                with sb.let(kernel_var, cast(p_kernels[node2kernel[idx]], kernel_type)):
                    sb += kernel_var(*node_params)

                # Free memory if reached last usage of Tensor
                for x in node.inputs:
                    t_mapping.dec_usage_count(x)
                    if t_mapping.get_usage_count(x) == 0 and t_mapping.is_local(x):
                        sb += memory_planner_free(d2i[x.device.kind], t_mapping.get_var(x))
                        t_mapping.set_allocated(x, False)

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

    # save kernels
    src_list = [compiled_task.task_dir for compiled_task in cgraph.compiled_tasks]
    dst_list = [os.path.join(cache_dir, 'kernels/{}'.format(i)) for i, _ in enumerate(cgraph.compiled_tasks)]
    # save graph module
    src_list.append(cgraph.graph_module.module_dir)
    dst_list.append(os.path.join(cache_dir, 'graph_module/'))
    copy_tree_ignore_existing(src_list, dst_list)
    # save weights
    with open(os.path.join(cache_dir, 'weights.npz'), 'wb') as f:
        numpy.savez(f, *[weight.cpu().numpy() for weight in cgraph.weights])
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


# Supporting storage for Tensor2VarMap
class Tensor2VarMapUnit:
    def __init__(self, v: Var, local: bool, usage_count=0):
        self.var = v
        self.local = local
        self.usage_count = usage_count


# This class provide info about correspondence between all Tensors in graph
# and variables allocated to store these Tensors.
# Also it stores info about liveness of Tensers.
# Also store info allocated variable or not.
# Used for implementation of launch function in the compiled graph.
class Tensor2VarMap:
    def __init__(
        self,
        in_tensors: List[Tensor],
        in_vars: List[Var],
        out_tensors: List[Tensor],
        out_vars: List[Var],
        w_tensors: List[Tensor],
        w_vars: List[Var],
        v_tensors: List[Tensor],
        v_vars: List[Var],
        usage_count: Dict[Tensor, int],
        cpu_base: Var,
        cuda_base: Var,
    ):

        in_map: Dict[Tensor, Tensor2VarMapUnit] = {
            x: Tensor2VarMapUnit(in_vars[i], False) for i, x in enumerate(in_tensors)
        }
        out_map: Dict[Tensor, Tensor2VarMapUnit] = {
            x: Tensor2VarMapUnit(out_vars[i], False) for i, x in enumerate(out_tensors)
        }
        w_map: Dict[Tensor, Tensor2VarMapUnit] = {
            x: Tensor2VarMapUnit(w_vars[i], False) for i, x in enumerate(w_tensors)
        }
        v_map: Dict[Tensor, Tensor2VarMapUnit] = {
            x: Tensor2VarMapUnit(v_vars[i], True) for i, x in enumerate(v_tensors)
        }
        self.map: Dict[Tensor, Var] = {**in_map, **out_map, **w_map, **v_map}
        # Init usage count
        for x in usage_count:
            self.map[x].usage_count = usage_count[x]
        self.cpu_base = cpu_base
        self.cuda_base = cuda_base
        # Here we store a info about whether variable is allocated or not
        self.is_var_allocated: Dict[Var, bool] = {self.map[i].var: False for i in self.map}

    def dec_usage_count(self, x: Tensor):
        self.map[x].usage_count -= 1

    def inc_usage_count(self, x: Tensor):
        self.map[x].usage_count += 1

    def get_usage_count(self, x: Tensor) -> int:
        return self.map[x].usage_count

    def is_local(self, x: Tensor) -> bool:
        return self.map[x].local

    def is_global(self, x: Tensor) -> bool:
        return not self.map[x].local

    def get_var(self, x: Tensor) -> Var:
        return self.map[x].var

    def get_full_addr(self, x: Tensor):
        d2b = {'cpu': self.cpu_base, 'cuda': self.cuda_base}
        if self.is_local(x):
            return d2b[x.device.kind] + self.get_var(x)
        else:
            return self.get_var(x)

    def is_allocated(self, x: Tensor):
        v = self.get_var(x)
        return self.is_var_allocated[v]

    def set_allocated(self, x: Tensor, value):
        v = self.get_var(x)
        self.is_var_allocated[v] = value

    # Implement share map optimisation. Share map means that input and output tensors should be the same.
    # In below description: input - graph input tensor; output - graph output tensor; xi - graph intermediate tensor
    # 1. if op is intput -> x1, then change it on input -> input
    # 2. if op is x1 -> output, then change it on output -> output
    # 3. if op is x1 -> x2, then change it on x1 -> x1
    # Need "while change" loop to pop up output as high as possible.
    # Note: "while change" algorithms might be changed on two passes algoithm.
    # The first top-down, the second down-top. Does not a lot of sense here -
    # it is not critical code from time of work point of view.
    def process_share_map(self, graph_nodes: List[Operator]):
        change = True
        while change:
            change = False
            for node in graph_nodes:
                if node.share_map:
                    for i in node.share_map:
                        in_tensor = node.inputs[i]
                        out_tensor = node.outputs[node.share_map[i]]
                        if self.is_global(in_tensor) and self.is_global(out_tensor):
                            # Nothing can do. memory for input and output of graph is allocated on
                            # higher level. Theoreticaly it's possible to process that on higher
                            # level, but this case should be rare
                            continue
                        if self.is_global(in_tensor):
                            if self.map[out_tensor].var is not self.map[in_tensor].var:
                                self.map[out_tensor].var = self.map[in_tensor].var
                                self.dec_usage_count(out_tensor)
                                self.inc_usage_count(in_tensor)
                                self.map[out_tensor].local = False
                                change = True
                        elif self.is_global(out_tensor):
                            if self.map[in_tensor].var is not self.map[out_tensor].var:
                                self.map[in_tensor].var = self.map[out_tensor].var
                                self.dec_usage_count(in_tensor)
                                self.inc_usage_count(out_tensor)
                                self.map[in_tensor].local = False
                                change = True
                        else:
                            if self.map[out_tensor].var is not self.map[in_tensor].var:
                                self.map[out_tensor].var = self.map[in_tensor].var
                                self.dec_usage_count(out_tensor)
                                self.inc_usage_count(in_tensor)
                                change = True
