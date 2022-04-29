from .base import GraphPass, PassContext, logger
from hidet.ir.expr import Var, var, TensorElement
from hidet.tos.ir import FlowGraph, Operator
from hidet.ir.dialects.compute import TensorNode, GridCompute
from hidet.tos.ir.functors import clone, analyze_usage
from hidet.ir.task import Task, Epilogue, is_elementwise, is_unary_elementwise
from hidet.ir.functors import rewrite, collect

from .common import concat_op_name


class FuseUnaryEpiloguePass(GraphPass):

    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        graph = clone(graph)
        usage = analyze_usage(graph)
        graph.update_nodes()

        while True:
            success = False
            for u_op in graph.nodes:
                if len(u_op.task.inverse_map) == 0:
                    continue
                if len(u_op.task.prologues) + len(u_op.task.epilogues) > 0:
                    continue
                if not is_unary_elementwise(u_op):
                    continue
                if len(usage[u_op.inputs[0]]) > 1:
                    # intermediate tensor can not be used by other operators.
                    continue
                v_op, out_idx = u_op.inputs[0].trace
                if v_op is None:
                    continue

                # v_op -> u_op  ==> v_op (with u_op as epilogue)
                assert isinstance(v_op, Operator)
                v_task = v_op.task
                u_task = u_op.task
                u_output = u_task.outputs[0]
                v_output = v_task.outputs[out_idx]
                parameters = v_task.parameters.copy()

                # fetch reverse map
                imap = u_op.task.inverse_map[u_op.task.inputs[0]]

                existed_epilogue = v_task.epilogues[v_output] if v_output in v_task.epilogues else None
                # input indices for the input of u task, use existed epilogue's when possible
                if existed_epilogue:
                    input_indices = existed_epilogue.out_indices
                else:
                    input_indices = [var('i') for _ in range(len(u_op.inputs[0].shape))]

                # prepare the indices for the output of u task
                rmap = {a: b for a, b in zip(imap.axes, input_indices)}
                dest_indices = [rewrite(dest_index_expr, rmap) for dest_index_expr in imap.indices]

                # input value for the input of u task, use existed epilogue's when possible
                if existed_epilogue:
                    input_value = existed_epilogue.value
                else:
                    input_value = Var('orig_value', v_output.data_type.scalar_type)

                # prepare the TensorElement in the u task's expression
                grid_compute = u_output.grid_compute
                tensor_elements = [te for te in collect(grid_compute.value, TensorElement) if te.base is u_task.inputs[0]]
                if len(tensor_elements) == 0:
                    raise ValueError('Encountered a task whose output has not accessed its input.')
                if len(tensor_elements) > 1:    # accessed input twice, we do not fuse in this case
                    continue
                te = tensor_elements.pop()

                # prepare the value of u task's output
                rmap = {a: b for a, b in zip(grid_compute.axes, dest_indices)}
                rmap[te] = input_value
                value = rewrite(grid_compute.value, rmap)

                # prepare the parameters and epilogue
                new_output_node = TensorNode(u_output.name, u_output.data_type)
                if existed_epilogue:
                    existed_output_in_param = existed_epilogue.out_tensor
                else:
                    existed_output_in_param = v_output
                parameters = [p if p is not existed_output_in_param else new_output_node for p in parameters]
                epilogue = Epilogue(
                    extra_inputs=existed_epilogue.extra_inputs if existed_epilogue else [],
                    indices=input_indices,
                    orig_value=input_value,
                    value=value,
                    out_indices=dest_indices,
                    out_tensor=new_output_node
                )

                # prepare the fused op
                epilogues = v_task.epilogues.copy()
                epilogues.update({v_output: epilogue})
                outputs = v_op.outputs[:out_idx] + [u_op.outputs[0]] + v_op.outputs[out_idx + 1:]

                fused_op = Operator(
                    inputs=v_op.inputs,
                    task=Task(
                        name=v_task.name,
                        inputs=v_task.inputs,
                        outputs=v_task.outputs,
                        prologues=v_task.prologues,
                        epilogues=epilogues,
                        parameters=parameters
                    ),
                    outputs=outputs,
                    name=concat_op_name(v_op.name, u_op.name),
                    **v_op.attributes,
                    **u_op.attributes
                )
                fused_op.outputs[out_idx].trace = (fused_op, 0)

                # update graph.nodes
                graph.nodes = [node if node is not u_op else fused_op for node in graph.nodes if node is not v_op]
                success = True

                if PassContext.current().verbose:
                    logger.info('Fused {} => {}'.format([v_op.task.name, u_op.task.name], fused_op.task.name))

            if not success:
                break
        return graph


def fuse_unary_epilogue_pass() -> GraphPass:
    return FuseUnaryEpiloguePass()
