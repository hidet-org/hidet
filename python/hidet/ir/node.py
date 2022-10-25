from typing import Mapping, Type, Any, List


class Node:
    _dispatch_index = {None: 0}

    def __str__(self):
        from hidet.ir.functors.printer import astext    # pylint: disable=import-outside-toplevel
        return astext(self)

    def __repr__(self):
        return str(self)

    def __int__(self):
        return None

    @classmethod
    def class_index(cls):
        if not hasattr(cls, '_class_index'):
            setattr(cls, '_class_index', len(Node._dispatch_index))
            Node._dispatch_index[cls] = getattr(cls, '_class_index')
        return getattr(cls, '_class_index')

    @staticmethod
    def dispatch_table(mapping: Mapping[Type['Node'], Any]) -> List[Any]:
        table = []
        for cls, target in mapping.items():
            idx = cls.class_index()
            while idx >= len(table):
                table.append(None)
            table[idx] = target
        return table
