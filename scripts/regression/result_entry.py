import os
import json

class ResultEntry:
    def __init__(self, shape, dtype, latency, attrs=None) -> None:
        self.shape = shape
        self.dtype = dtype
        self.latency = latency
        self.attrs = attrs
    
    def __str__(self) -> str:
        s = f"{f'Shape: {self.shape}': <30}{f'dtype: {self.dtype}': <20}{f'latency: {self.latency}ms': <15}"
        return s


class ResultGroup:
    def __init__(self, name: str, device_name: str) -> None:
        self.result_entries = []
        self.name = name
        self.device_name = device_name
    
    def __str__(self) -> str:
        s = self.name + '\n'
        for entry in self.result_entries:
            s += '\t' + str(entry) + '\n'
        return s
    
    def add_entry(self, entry: ResultEntry) -> None:
        self.result_entries.append(entry)

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