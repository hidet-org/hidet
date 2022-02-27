from typing import Callable, MutableMapping, Iterator, Sequence
import itertools
import contextlib
import os
import time


def prod(seq: Sequence):
    if len(seq) == 0:
        return 1
    else:
        c = seq[0]
        for i in range(1, len(seq)):
            c = c * seq[i]
        return c


class Timer:
    def __init__(self, msg=None, file=None, verbose=True):
        self.start_time = None
        self.end_time = None
        self.msg = msg
        self.verbose = verbose
        self.file = file

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.msg is not None and self.verbose:
            print('{} {:.3f} seconds'.format(self.msg, self.elapsed_seconds()), file=self.file)

    def elapsed_seconds(self) -> float:
        return self.end_time - self.start_time


class DictCustomKey(MutableMapping, dict):
    def __init__(self, hash_func: Callable[[object], int]):
        super().__init__()
        self.hash_func = hash_func

    def __delitem__(self, v):
        return dict.__delitem__(self, self.hash_func(v))

    def __len__(self) -> int:
        return dict.__len__(self)

    def __iter__(self):
        return dict.__iter__(self)

    def __getitem__(self, item):
        return dict.__getitem__(self, self.hash_func(item))

    def __setitem__(self, key, value):
        return dict.__setitem__(self, self.hash_func(key), value)


def repeat_until_converge(func, obj, limit=None):
    i = 0
    while True:
        i += 1
        orig_obj = obj
        obj = func(obj)
        if obj is orig_obj:
            return obj
        if limit is not None and i >= limit:
            return obj


class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_next_file_index(dirname: str) -> int:
    indices = set()
    for fname in os.listdir(dirname):
        parts = fname.split('_')
        with contextlib.suppress(ValueError):
            indices.add(int(parts[0]))
    for idx in itertools.count(0):
        if idx not in indices:
            return idx
