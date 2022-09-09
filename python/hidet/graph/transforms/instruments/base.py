from hidet.graph.ir.flow_graph import FlowGraph


class GraphPassInstrument:
    def before_all_passes(self, graph: FlowGraph):
        pass

    def before_pass(self, pass_name: str, graph: FlowGraph):
        pass

    def after_pass(self, pass_name: str, graph: FlowGraph):
        pass

    def after_all_passes(self, graph: FlowGraph):
        pass




