from typing import Any, Union, List, Dict, Tuple
from hidet.ir.node import Node
from hidet.utils import same_list


class BaseFunctor:
    def __init__(self, use_memo=True):
        self.memo = {} if use_memo else None

    def __call__(self, node: Any):
        return self.visit(node)

    def visit(self, node: Union[Node, Tuple, List, Dict[str, Any], str, int, float]):
        key = id(node) if isinstance(node, (list, dict)) else node
        if self.memo is not None and key in self.memo:
            return self.memo[key]

        # iterate through the mro of the class to find a visit_dispatch method
        for cls in type(self).__mro__:
            if cls is object:
                # we have reached the end of the mro but still not found a visit_dispatch method to dispatch the node
                raise NotImplementedError("Can not dispatch object with type {}".format(type(node)))
            if 'visit_dispatch' not in cls.__dict__:
                continue
            ret = cls.visit_dispatch(self, node)
            if ret is not NotImplemented:
                break

        if self.memo is not None:
            self.memo[key] = ret
        #
        # from hidet.ir.functors import IRRewriter
        # if isinstance(self, IRRewriter) and ret is not node:
        #     print('Rewrite[{}]: {} -> {}'.format('*' if node is not ret else ' ', node, ret))
        #
        return ret

    def visit_dispatch(self, node: Union[Node, Tuple, List, Dict[str, Any], str, int, float]):
        if isinstance(node, tuple):
            return self.visit_Tuple(node)
        elif isinstance(node, list):
            return self.visit_List(node)
        elif isinstance(node, dict):
            return self.visit_Dict(node)
        elif isinstance(node, (str, int, float)) or node is None:
            return self.visit_PyConstant(node)
        elif isinstance(node, Node):
            return self.visit_NotDispatchedNode(node)
        else:
            return NotImplemented

    def visit_Tuple(self, tp: Tuple):
        raise NotImplementedError()

    def visit_List(self, lst: List):
        raise NotImplementedError()

    def visit_Dict(self, d: Dict):
        raise NotImplementedError()

    def visit_NotDispatchedNode(self, n: Node):
        raise NotImplementedError()

    def visit_PyConstant(self, c: Union[str, int, float, None]):
        raise NotImplementedError()


class BaseVisitor(BaseFunctor):
    def visit_Tuple(self, t: Tuple):
        for v in t:
            self.visit(v)

    def visit_List(self, l: List):
        for v in l:
            self.visit(v)

    def visit_Dict(self, d: Dict):
        for v in d.values():
            self.visit(v)

    def visit_NotDispatchedNode(self, n: Node):
        pass

    def visit_PyConstant(self, c: Union[str, int, float, None]):
        pass


class BaseRewriter(BaseFunctor):
    def visit_Tuple(self, tp: Tuple):
        updated = tuple(self.visit(v) for v in tp)
        return tp if same_list(updated, tp) else updated

    def visit_List(self, lst: List):
        updated = [self.visit(v) for v in lst]
        return lst if same_list(updated, lst) else updated

    def visit_Dict(self, d: Dict):
        if any(not isinstance(k, str) for k in d.keys()):
            raise NotImplementedError("Can not dispatch dict with non-str key")
        updated = {k: self.visit(v) for k, v in d.items()}
        return d if same_list(updated.values(), d.values()) else updated

    def visit_NotDispatchedNode(self, n: Node):
        return n

    def visit_PyConstant(self, c: Union[str, int, float, None]):
        return c
