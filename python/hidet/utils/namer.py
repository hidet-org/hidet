from collections import defaultdict


class Namer:
    def __init__(self):
        self.name_id_clock = defaultdict(int)
        self.obj_name = {}
        self.clear()

    def clear(self):
        self.name_id_clock.clear()
        self.obj_name.clear()
        # add keywords in target language
        keywords = [
            'const'
        ]
        for kw in keywords:
            self.name_id_clock[kw] = 0

    def get_name(self, e, hint=None):
        from hidet.ir.expr import Var
        from hidet.ir.dialects.compute import ScalarNode, TensorNode
        if e in self.obj_name:
            return self.obj_name
        if hint:
            orig_name = hint
        elif isinstance(e, Var) and e.hint is not None:
            orig_name = e.hint
        elif isinstance(e, (ScalarNode, TensorNode)):
            orig_name = e.name
        else:
            alias = {
                ScalarNode: 'scalar',
                TensorNode: 'tensor',
                Var: 'v',
            }
            orig_name = alias[type(e)] if type(e) in alias else type(e).__name__

        if orig_name in self.name_id_clock:
            name = orig_name
            while name in self.name_id_clock:
                self.name_id_clock[orig_name] += 1
                name = orig_name + '_' + str(self.name_id_clock[orig_name])
        else:
            self.name_id_clock[orig_name] = 0
            name = orig_name

        self.obj_name[e] = name
        return name
