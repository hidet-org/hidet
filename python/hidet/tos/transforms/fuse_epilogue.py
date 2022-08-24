from .base import GraphPass, PassContext, logger
from hidet.ir.expr import Var, var, TensorElement
from hidet.tos.ir import FlowGraph, Operator
from hidet.ir.dialects.compute import TensorNode, GridCompute, ScalarNode
from hidet.tos.ir.functors import clone, analyze_usage
from hidet.ir.task import Task, Epilogue, is_injective_task, is_unary_injective_task, is_elementwise_task
from hidet.ir.functors import rewrite, collect
from hidet.utils import py
from .utils import is_barrier


def try_fuse(graph: FlowGraph, usage) -> bool:
    for u_op in graph.nodes:
        if len(u_op.task.prologues) + len(u_op.task.epilogues) > 0:
            continue
        if is_barrier(u_op):
            continue
        if not is_injective_task(u_op.task) or len(u_op.outputs) != 1:
            # u_op must be an elementwise op with a single output (may have multiple inputs)
            continue
        for u_input_idx, u_input in enumerate(u_op.inputs):
            if u_input.trace is None:
                # skip graph input and const tensor
                continue
            if len(usage[u_input]) > 1:
                # intermediate tensor can not be used by other operators.
                continue
            v_op, v_output_index = u_input.trace
            if v_op is None:
                continue
            assert isinstance(v_op, Operator)
            if is_barrier(v_op):
                continue

            if is_injective_task(v_op.task):
                # we do not fuse an injective_op v with an injective_op u by taking u as epilogue.
                continue

            # v_op -> u_op  ==> v_op (with u_op as epilogue)
            v_task = v_op.task
            u_task = u_op.task
            u_output = u_task.outputs[0]
            v_output = v_task.outputs[v_output_index]
            u_task_input = u_task.inputs[u_input_idx]

            if u_task_input not in u_task.inverse_map:
                # u_op must be invertible regards to enumerated input that wants to be fused along.
                continue

            parameters = v_task.parameters.copy()

            # fetch reverse map
            imap = u_op.task.inverse_map[u_task_input]

            existed_epilogue = v_task.epilogues[v_output] if v_output in v_task.epilogues else None
            # input indices for the input of u task, use existed epilogue's when possible
            if existed_epilogue:
                indices = existed_epilogue.indices
                rmap = {a: b for a, b in zip(imap.axes, existed_epilogue.out_indices)}
            else:
                indices = [var('i') for _ in range(len(u_input.shape))]
                rmap = {a: b for a, b in zip(imap.axes, indices)}
            dest_indices = [rewrite(dest_index_expr, rmap) for dest_index_expr in imap.indices]

            # prepare the TensorElement in the u task's expression
            grid_compute = u_output.grid_compute
            tensor_elements = [te for te in collect(grid_compute.value, TensorElement) if te.base is u_task_input]
            if len(tensor_elements) == 0:
                raise ValueError('Encountered a task whose output has not accessed its input.')
            if len(tensor_elements) > 1:    # accessed input twice, we do not fuse in this case
                continue
            te = tensor_elements.pop()

            # prepare the value of u task's output
            rmap = {a: b for a, b in zip(grid_compute.axes, dest_indices)}
            if existed_epilogue:
                orig_value = existed_epilogue.orig_value
                rmap[te] = existed_epilogue.value
            else:
                orig_value = Var('orig_value', v_output.data_type.scalar_type)
                rmap[te] = orig_value
            value = rewrite(grid_compute.value, rmap)

            # prepare the parameters and epilogue
            new_output_node_value = rewrite(grid_compute.value, {u_task_input: v_output})
            new_output_node = TensorNode(
                name=u_output.name,
                data_type=u_output.data_type,
                grid_compute=GridCompute(
                    input_tensors=collect(new_output_node_value, TensorNode, stop_when_found=True),
                    input_scalars=collect(new_output_node_value, ScalarNode, stop_when_found=True),
                    shape=grid_compute.shape,
                    axes=grid_compute.axes,
                    value=new_output_node_value
                )
            )
            input_params = parameters[:len(v_op.inputs)]
            output_params = parameters[len(v_op.inputs):]
            v_task_extra_inputs = [input for input in u_task.inputs if input is not u_task_input]
            v_extra_inputs = [input_tensor for input_tensor in u_op.inputs if input_tensor is not u_input]
            input_params.extend(v_task_extra_inputs)
            if existed_epilogue:
                existed_output_in_param = existed_epilogue.out_tensor
                extra_inputs = existed_epilogue.extra_inputs + v_task_extra_inputs
            else:
                existed_output_in_param = v_output
                extra_inputs = v_task_extra_inputs
            output_params = [p if p is not existed_output_in_param else new_output_node for p in output_params]
            parameters = input_params + output_params
            epilogue = Epilogue(
                extra_inputs=extra_inputs,
                indices=indices,
                orig_value=orig_value,
                value=value,
                out_indices=dest_indices,
                out_tensor=new_output_node
            )

            # prepare the fused op
            epilogues = v_task.epilogues.copy()
            epilogues.update({v_output: epilogue})
            outputs = v_op.outputs[:v_output_index] + [u_op.outputs[0]] + v_op.outputs[v_output_index + 1:]

            task = v_task.copy()
            task.name = '{}_{}'.format(v_task.name, u_task.name)
            task.epilogues = epilogues
            task.parameters = parameters

            fused_op = Operator(
                inputs=v_op.inputs + v_extra_inputs,
                task=task,
                outputs=outputs,
                name=v_op.name,
                attributes=v_op.attrs
            )
            fused_op.outputs[v_output_index].trace = (fused_op, 0)

            # update graph.nodes
            graph.nodes = [node if node is not u_op else fused_op for node in graph.nodes if node is not v_op]

            if PassContext.current().configs['verbose']:
                logger.info('Fused epilogue {} {}'.format(py.color_text(v_op.name, idx=1), py.color_text(u_op.name, idx=2)))
                logger.debug('{}'.format(task))
            return True
    return False


class FuseEpiloguePass(GraphPass):

    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        graph = clone(graph)
        usage = analyze_usage(graph)
        graph.update_nodes()

        while True:
            success = try_fuse(graph, usage)
            if not success:
                break
        return graph


def fuse_epilogue_pass() -> GraphPass:
    return FuseEpiloguePass()
