from typing import Callable, MutableMapping, Iterator
import time


def prod(seq: list):
    if len(seq) == 0:
        return 1
    else:
        c = seq[0]
        for i in range(1, len(seq)):
            c = c * seq[i]
        return c


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

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
