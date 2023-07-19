import os
import json
import numpy as np

class ResultEntry:
    def __init__(self, shape, dtype, latency, ref_latency, attrs=None) -> None:
        self.shape = shape
        self.dtype = dtype
        self.latency = latency
        self.attrs = attrs
        self.ref_latency = ref_latency
        self.speedup = None
        if not np.allclose(latency, ref_latency, rtol=0.08):
            self.speedup = ref_latency / latency
    
    def __str__(self) -> str:
        s = f"{f'Shape: {self.shape}': <28}{f'dtype: {self.dtype}': <17}{f'latency: {self.latency:.5f}ms': <23}"
        if self.speedup is not None:
            s2 = f"{f'chg: {(self.speedup - 1.0) * 100:.2f}%': <15}"
            s = s + s2
        return s


class ResultGroup:
    def __init__(self, name: str, device_name: str) -> None:
        self.result_list = []
        self.name = name
        self.device_name = device_name
        self.speedup_list = []
        self.slowdown_list = []
    
    def __str__(self) -> str:
        s = self.name + '\n'
        if len(self.speedup_list) > 0:
            s += ' '*4 +  '*****SPEEDUP*****:\n'
            for entry in self.speedup_list:
                s += ' '*6 + str(entry) + '\n'
            s += '\n'
        if len(self.slowdown_list) > 0:
            s += ' '*4 + '*****SLOWDOWN*****:\n'
            for entry in self.slowdown_list:
                s += ' '*6 + str(entry) + '\n'
            s += '\n'
        for entry in self.result_list:
            s += ' '*4 + str(entry) + '\n'
        return s
    
    def add_entry(self, entry: ResultEntry) -> None:
        if entry.speedup is not None:
            speedup = entry.speedup > 1.0
            if speedup:
                self.speedup_list.append(entry)
            else:
                self.slowdown_list.append(entry)
        else:
            self.result_list.append(entry)


def load_regression_data() -> dict:
    if os.path.exists('./scripts/regression/regression_data.json'):
        data_file = './scripts/regression/regression_data.json'
    elif os.path.exists('./regression_data.json'):
        data_file = './regression_data.json'
    else:
        raise FileNotFoundError("regression_data.json not found. Please run "
                    "this script from the root directory of the repository "
                    "or $ROOT/scripts/regression/")
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data