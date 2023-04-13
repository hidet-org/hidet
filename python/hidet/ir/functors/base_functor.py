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
from typing import Any, Union, List, Dict, Tuple, Type
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

        functor_cls: Type[BaseFunctor] = type(self)
        node_cls = type(node)
        dispatch_table = getattr(functor_cls, '__dispatch_table', None)
        if dispatch_table and node_cls in dispatch_table:
            # fast path
            ret = dispatch_table[node_cls](self, node)
        else:
            # slow path
            # iterate through the mro of the class to find a visit_dispatch method that can handle the node
            for cls in type(self).__mro__:
                dispatch_func = cls.__dict__.get('visit_dispatch', None)  # do not use getattr here
                if dispatch_func is None:
                    continue
                ret = dispatch_func(self, node)
                if ret is not NotImplemented:
                    # record the dispatch function to the dispatch table of the functor class for fast path
                    if dispatch_table is None:
                        dispatch_table = {}
                        setattr(functor_cls, '__dispatch_table', dispatch_table)
                    dispatch_table[node_cls] = dispatch_func
                    break
            else:
                raise NotImplementedError("Can not dispatch object with type {}".format(type(node)))

        if self.memo is not None:
            self.memo[key] = ret

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
