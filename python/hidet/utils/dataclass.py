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
# pylint: disable=unused-import
from dataclasses import fields, is_dataclass, asdict


def from_dict(cls, data):
    # pylint: disable=protected-access
    if is_dataclass(cls):
        args = []
        if not isinstance(data, dict):
            raise TypeError(f'expected dict, got {type(data)}')
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
            key_type, value_type = cls.__args__
            return {key_type(k): from_dict(value_type, v) for k, v in data.items()}
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
