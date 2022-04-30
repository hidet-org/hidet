from .base import GraphPass, PassContext, logger
from hidet.tos.ir import FlowGraph, Operator
from hidet.tos.ir.functors import clone, analyze_usage
from hidet.ir.task import Task, is_unary_elementwise
from hidet.ir.functors import rewrite
from hidet.utils import py

from .common import concat_op_name


class FuseUnaryElementwise(GraphPass):
    """
    Fuse all consecutive unary elementwise operators together.
    """
    def process_graph(self, graph: FlowGraph) -> FlowGraph:
        graph = clone(graph)
        usage = analyze_usage(graph)
        graph.update_nodes()

        while True:
            success = False
            for u_op in graph.nodes:
                if not is_unary_elementwise(u_op.task):
                    continue
                if len(usage[u_op.inputs[0]]) > 1:
                    # intermediate tensor can not be used by other operators.
                    continue
                v_op = u_op.inputs[0].op
                if v_op is None:
                    continue
                if not is_unary_elementwise(v_op.task):
                    continue

                # create fused op
                # x --(v_op)--> y --(u_op)--> z
                x = v_op.task.inputs[0]
                y = v_op.task.outputs[0]
                z = rewrite(u_op.task.outputs[0], {u_op.task.inputs[0]: y})
                if v_op.task.inverse_map and u_op.task.inverse_map:
                    inverse_map = {x: list(v_op.task.inverse_map.values())[0] + list(u_op.task.inverse_map.values())[0]}
                else:
                    inverse_map = None
                fused_op = Operator(
                    inputs=v_op.inputs,
                    task=Task(
                        name='{}_{}'.format(v_op.task.name, u_op.task.name),
                        inputs=[x],
                        outputs=[z],
                        inverse_map=inverse_map
                    ),
                    outputs=u_op.outputs,
                    name=concat_op_name(v_op.name, u_op.name),
                    **{**v_op.attributes, **u_op.attributes},
                )
                fused_op.outputs[0].trace = (fused_op, 0)

                # update graph.nodes
                graph.nodes = [node if node is not u_op else fused_op for node in graph.nodes if node is not v_op]
                success = True

                # log
                if PassContext.current().verbose:
                    logger.info('Fused elementwise {} {}'.format(py.color_text(v_op.name, idx=1), py.color_text(u_op.name, idx=2)))
            if not success:
                break
        return graph


def fuse_unary_elementwise_pass() -> GraphPass:
    return FuseUnaryElementwise()
