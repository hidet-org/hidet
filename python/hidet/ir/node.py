from typing import Mapping, Type, Any, List


class Node:
    _dispatch_index = {}

    def __str__(self):
        from hidet.ir.functors.printer import astext
        return astext(self)

    def __repr__(self):
        return str(self)

    def __int__(self):
        return None

    @classmethod
    def class_index(cls):
        if not hasattr(cls, '_class_index'):
            setattr(cls, '_class_index', len(Node._dispatch_index))
        return getattr(cls, '_class_index')

    def dispatch_table(self, mapping: Mapping[Type['Node'], Any]) -> List[Any]:
        table = []
        for cls, target in mapping.items():
            idx = cls.class_index()
            while idx <= len(table):
                table.append(None)
            table[idx] = target
        return table

