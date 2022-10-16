from typing import TypeVar, Iterable, Tuple, List, Union
import numpy as np
import cProfile
import contextlib
import io
import itertools
import os
import pstats
import time
from tabulate import tabulate
from typing import Callable, MutableMapping, Sequence


def unique(seq: Sequence) -> List:
    added = set()
    return [item for item in seq if (item not in added and not added.add(item))]


def prod(seq: Sequence):
    if len(seq) == 0:
        return 1
    else:
        c = seq[0]
        for i in range(1, len(seq)):
            c = c * seq[i]
        return c


TypeA = TypeVar('TypeA')
TypeB = TypeVar('TypeB')


def strict_zip(a: Sequence[TypeA], b: Sequence[TypeB]) -> Iterable[Tuple[TypeA, TypeB]]:
    if len(a) != len(b):
        raise ValueError('Expect two sequence have the same length in zip, got length {} and {}.'.format(len(a), len(b)))
    return zip(a, b)


class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    MAGENTA = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def green(v, fmt='{}'):
    return COLORS.OKGREEN + fmt.format(v) + COLORS.ENDC


def cyan(v, fmt='{}'):
    return COLORS.OKCYAN + fmt.format(v) + COLORS.ENDC


def blue(v, fmt='{}'):
    return COLORS.OKBLUE + fmt.format(v) + COLORS.ENDC


def red(v, fmt='{}'):
    return COLORS.FAIL + fmt.format(v) + COLORS.ENDC


def magenta(v, fmt='{}'):
    return COLORS.MAGENTA + fmt.format(v) + COLORS.ENDC


def bold(v, fmt='{}'):
    return COLORS.BOLD + fmt.format(v) + COLORS.ENDC


def color(v, fmt='{}', fg='default', bg='default'):
    fg_code = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "default": 39,
    }
    bg_code = {
        "black": 40,
        "red": 41,
        "green": 42,
        "yellow": 43,
        "blue": 44,
        "magenta": 45,
        "cyan": 46,
        "white": 47,
        "default": 49
    }
    return '\033[{};{}m{}\033[0m'.format(fg_code[fg], bg_code[bg], fmt.format(v))


def color_table():
    fg_names = ["default", "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    bg_names = ["default", "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    print('{:>10} {:>10}   {:<10}'.format('fg', 'bg', 'text'))
    for bg in bg_names:
        for fg in fg_names:
            print('{:>10} {:>10}   {}'.format(fg, bg, color('sample text', fg=fg, bg=bg)))


def color_rgb(v, fg, fmt='{}'):
    return '\033[38;2;{};{};{}m{}\033[0m'.format(fg[0], fg[1], fg[2], v)


def color_text(v, fmt='{}', idx: int = 0):
    if idx == 0:
        return fmt.format(v)
    colors = {
        1: (153, 96, 52),
        2: (135, 166, 73)
    }
    return color_rgb(v, colors[idx], fmt=fmt)


def nocolor(s: str) -> str:
    for name, value in COLORS.__dict__.items():
        if isinstance(value, str) and value[0] == '\033':
            s = s.replace(value, '')
    return s


def str_indent(msg: str, indent=0) -> str:
    lines = msg.split('\n')
    lines = [' ' * indent + line for line in lines]
    return '\n'.join(lines)


class Timer:
    def __init__(self, msg=None, file=None, verbose=True, stdout=True):
        self.start_time = None
        self.end_time = None
        self.msg = msg
        self.stdout = stdout
        self.verbose = verbose
        self.file = file

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.msg is not None and self.verbose:
            msg = '{} {}'.format(self.msg, green(self.time2str(self.end_time - self.start_time)))
            if self.stdout:
                print(msg)
            if self.file:
                if isinstance(self.file, str):
                    with open(self.file, 'w') as f:
                        f.write(nocolor(msg))
                else:
                    self.file.write(msg + '\n')

    def elapsed_seconds(self) -> float:
        return self.end_time - self.start_time

    def time2str(self, seconds: float) -> str:
        if seconds < 1:
            return '{:.1f} {}'.format(seconds * 1000, 'ms')
        elif seconds < 60:
            return '{:.1f} {}'.format(seconds, 'seconds')
        elif seconds < 60 * 60:
            return '{:.1f} {}'.format(seconds / 60, 'minutes')
        else:
            return '{:.1f} {}'.format(seconds / 60 / 60, 'hours')


# class DictCustomKey(MutableMapping, dict):
#     def __init__(self, hash_func: Callable[[object], int]):
#         super().__init__()
#         self.hash_func = hash_func
#
#     def __delitem__(self, v):
#         return dict.__delitem__(self, self.hash_func(v))
#
#     def __len__(self) -> int:
#         return dict.__len__(self)
#
#     def __iter__(self):
#         return dict.__iter__(self)
#
#     def __getitem__(self, item):
#         return dict.__getitem__(self, self.hash_func(item))
#
#     def __setitem__(self, key, value):
#         return dict.__setitem__(self, self.hash_func(key), value)
#

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


def same_list(lhs, rhs, use_equal=False):
    assert isinstance(lhs, (tuple, list)) and isinstance(rhs, (tuple, list))
    if len(lhs) != len(rhs):
        return False
    for l, r in zip(lhs, rhs):
        if use_equal:
            if l != r:
                return False
        else:
            if l is not r:
                return False
    return True


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


def line_profile():
    from line_profiler_pycharm import profile
    return profile


def initialize(*args, **kwargs):
    """
    Decorate an initialization function. After decorating with this function, the initialization function will be called
    after the definition.

    Parameters
    ----------
    args:
        The positional arguments of initializing.
    kwargs:
        The keyword arguments of initializing.

    Returns
    -------
    ret:
        A decorator that will call given function with args and kwargs,
        and return None (to prevent this function to be called again).
    """
    def decorator(f):
        f(*args, **kwargs)
        return None
    return decorator


def gcd(a: int, b: int) -> int:
    """
    Get the greatest common divisor of non-negative integers a and b.

    Parameters
    ----------
    a: int
        The lhs operand.
    b: int
        The rhs operand.

    Returns
    -------
    ret: int
        The greatest common divisor.
    """
    assert a >= 0 and b >= 0
    return a if b == 0 else gcd(b, a % b)


def lcm(a: int, b: int) -> int:
    """
    Get the least common multiple of non-negative integers a and b.
    Parameters
    ----------
    a: int
        The lhs operand.
    b: int
        The rhs operand.

    Returns
    -------
    ret: int
        The least common multiple.
    """
    return a // gcd(a, b) * b


def error_tolerance(a: Union[np.ndarray, 'Tensor'], b: Union[np.ndarray, 'Tensor']) -> float:
    """
    Given two tensors with the same shape and data type, this function finds the minimal e, such that

        abs(a - b) <= abs(b) * e + e

    Parameters
    ----------
    a: Union[np.ndarray, hidet.Tensor]
        The first tensor.
    b: Union[np.ndarray, hidet.Tensor]
        The second tensor.

    Returns
    -------
    ret: float
        The error tolerance between a and b.
    """
    from hidet.graph import Tensor
    if isinstance(a, Tensor):
        a = a.numpy()
    if isinstance(b, Tensor):
        b = b.numpy()
    if isinstance(a.dtype, np.floating):
        a = a.astype(np.float32)
        b = b.astype(np.float32)
    lf = 0.0
    rg = 9.0
    for step in range(20):
        mid = (lf + rg) / 2.0
        if np.allclose(a, b, rtol=mid, atol=mid):
            rg = mid
        else:
            lf = mid
    return (lf + rg) / 2.0


if __name__ == '__main__':
    # color_table()
    print(color_text('sample', idx=1))
    print(color_text('sample', idx=2))
