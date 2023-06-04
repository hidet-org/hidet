from dataclasses import fields, is_dataclass


def from_dict(cls, data):
    if is_dataclass(cls):
        args = []
        for field in fields(cls):
            args.append(from_dict(field.type, data[field.name]))
        return cls(*args)
    elif hasattr(cls, '__origin__'):
        name = str(cls.__origin__) if cls._name is None else 'typing.' + cls._name
        if name == 'typing.List':
            item_hint = cls.__args__[0]
            return [from_dict(item_hint, v) for v in data]
        elif name == 'typing.Tuple':
            if Ellipsis in cls.__args__:
                assert len(cls.__args__) == 2 and cls.__args__[1] == Ellipsis
                item_hint = cls.__args__[0]
                return tuple(from_dict(item_hint, v) for v in data)
            else:
                assert len(cls.__args__) == len(data)
                return tuple(from_dict(item_hint, v) for item_hint, v in zip(cls.__args__, data))
        elif name == 'typing.Dict':
            _, value_type = cls.__args__
            return {k: from_dict(value_type, v) for k, v in data.items()}
        elif name == 'typing.Union':
            return data
        else:
            print(cls)
            raise NotImplementedError(cls._name, cls)
    elif cls in [int, float, str]:
        return cls(data)
    elif isinstance(cls, str):
        raise NotImplementedError('Currently not support from __future__ import annotations')
    else:
        raise TypeError(f'unsupported type {cls}')
