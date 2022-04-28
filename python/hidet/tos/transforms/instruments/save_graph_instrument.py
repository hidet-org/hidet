import os

from hidet import utils
from hidet.tos.ir.graph import FlowGraph

from .base import GraphPassInstrument


class SaveGraphInstrument(GraphPassInstrument):
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.index = 0
        os.makedirs(out_dir, exist_ok=True)

    def before_all_passes(self, graph: FlowGraph):
        # first clean all json starting with indices
        for fname in os.listdir(self.out_dir):
            fpath = os.path.join(self.out_dir, fname)
            parts = fname.split('_')
            if os.path.isfile(fpath) and len(parts) > 1 and parts[0].isdigit() and fname.endswith('.json'):
                os.remove(fpath)
        with open(os.path.join(self.out_dir, '0_Origin.json'), 'w') as f:
            utils.netron.dump(graph, f)
            self.index += 1

    def after_pass(self, pass_name: str, graph: FlowGraph):
        with open(os.path.join(self.out_dir, '{}_{}.json'.format(self.index, pass_name)), 'w') as f:
            utils.netron.dump(graph, f)
            self.index += 1
