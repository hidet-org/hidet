import cProfile
import contextlib
import io
import itertools
import os
import pstats
import time
from tabulate import tabulate
from typing import Callable, MutableMapping, Sequence


def prod(seq: Sequence):
    if len(seq) == 0:
        return 1
    else:
        c = seq[0]
        for i in range(1, len(seq)):
            c = c * seq[i]
        return c


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
            print('{} {}{:.3f}{} seconds'.format(self.msg, COLORS.OKGREEN, self.elapsed_seconds(), COLORS.ENDC), file=self.file)

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


def get_next_file_index(dirname: str) -> int:
    indices = set()
    for fname in os.listdir(dirname):
        parts = fname.split('_')
        with contextlib.suppress(ValueError):
            indices.add(int(parts[0]))
    for idx in itertools.count(0):
        if idx not in indices:
            return idx


def factor(n):
    """
    example:
    factor(12) => [1, 2, 3, 4, 6, 12]
    """
    i = 1
    ret = []
    while i * i <= n:
        if n % i == 0:
            ret.append(i)
            if i * i != n:
                ret.append(n // i)
        i += 1
    return list(sorted(ret))


class HidetProfiler:
    def __init__(self, display_on_exit=True):
        self.pr = cProfile.Profile()
        self.display_on_exit = display_on_exit

    def __enter__(self):
        self.pr.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.disable()
        if self.display_on_exit:
            print(self.result())

    def result(self):
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        return str(s.getvalue())


class TableRowContext:
    def __init__(self, tb):
        self.tb = tb
        self.row = []

    def __iadd__(self, other):
        if isinstance(other, (tuple, list)):
            self.row.extend(other)
        else:
            self.row.append(other)

    def append(self, other):
        self.row.append(other)

    def extend(self, other):
        self.row.extend(other)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tb.rows.append(self.row)


class TableBuilder:
    def __init__(self, headers=tuple(), tablefmt='simple', floatfmt='.3f'):
        self.headers = list(headers)
        self.rows = []
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iadd__(self, row):
        self.rows.append(row)
        return self

    def __str__(self):
        return str(tabulate(self.rows, self.headers, tablefmt=self.tablefmt, floatfmt=self.floatfmt))

    def new_row(self) -> TableRowContext:
        return TableRowContext(tb=self)

    def extend_header(self, column_names):
        self.headers.extend(column_names)
