from typing import List, Sequence, Union
from hidet.ir.type import TypeNode


class TypeDecorator:
    def __init__(self, decorated_type: TypeNode, decorates: Sequence[str]):
        self.decorated_type: TypeNode = decorated_type
        self.decorates: List[str] = list(decorates)


def static(tp: Union[TypeNode, TypeDecorator]):
    if isinstance(tp, TypeNode):
        decorated_type = tp
        decorates = ['static']
    else:
        decorated_type = tp.decorated_type
        decorates = list(tp.decorates) + ['static']
    return TypeDecorator(decorated_type=decorated_type, decorates=decorates)
