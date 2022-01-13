from typing import Union, Optional, Sequence
import operator
import functools
from hidet.ir.node import Node


class SearchSpace(Node):
    def __init__(self, name):
        self.name = name

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()


class UnionSpace(SearchSpace):
    def __init__(self, name, sub_spaces: Optional[Sequence[SearchSpace]] = None):
        super().__init__(name)
        self.sub_spaces = list(sub_spaces) if sub_spaces else None

    def add_subspace(self, sub_space: SearchSpace):
        self.sub_spaces.append(sub_space)

    def __len__(self):
        return sum(len(space) for space in self.sub_spaces)

    def __getitem__(self, item):
        rank = len(self.sub_spaces)
        subspace_sizes = [len(s) for s in self.sub_spaces]
        for i in range(rank):
            if item < subspace_sizes[i]:
                return UnionChoice(self.name, self.sub_spaces[i][item])
            else:
                item -= subspace_sizes[i]
        raise IndexError()


class ProductSpace(SearchSpace):
    def __init__(self, name, sub_spaces: Optional[Sequence[SearchSpace]] = None):
        super().__init__(name)
        self.sub_spaces = list(sub_spaces) if sub_spaces else None

    def add_subspace(self, sub_space: SearchSpace):
        self.sub_spaces.append(sub_space)

    def __len__(self):
        return functools.reduce(operator.mul, [len(space) for space in self.sub_spaces])

    def __getitem__(self, item):
        rank = len(self.sub_spaces)
        subspace_sizes = [len(s) for s in self.sub_spaces]
        indices = [item // functools.reduce(operator.mul, subspace_sizes[i + 1:], 1) % subspace_sizes[i]
                   for i in range(rank)]
        choices = [self.sub_spaces[i][indices[i]] for i in range(rank)]
        return ProductChoice(self.name, choices)


class AtomSpace(SearchSpace):
    def __init__(self, name, choices):
        super().__init__(name)
        self.choices = choices

    def __len__(self):
        return len(self.choices)

    def __getitem__(self, item):
        return AtomChoice(self.name, self.choices[item])


class SpaceChoice(Node):
    def __init__(self, name):
        self.name = name


class UnionChoice(SpaceChoice):
    def __init__(self, name, choice):
        super().__init__(name)
        self.choice = choice


class ProductChoice(SpaceChoice):
    def __init__(self, name, choices):
        super().__init__(name)
        self.choices = choices
        self.name2choice = {choice.name: choice for choice in choices}

    def __getattr__(self, item):
        assert item in self.name2choice
        return self.name2choice[item]


class AtomChoice(SpaceChoice):
    def __init__(self, name, choice):
        super().__init__(name)
        self.choice = choice

