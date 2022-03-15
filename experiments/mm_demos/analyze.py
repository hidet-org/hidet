from collections import defaultdict
import numpy as np
import os
import os.path
import shutil
from tabulate import tabulate


class Data:
    def __init__(self, name2seq, key_names):
        self.key_names = key_names
        self.name2seq = name2seq
        self.key2latency = {}
        self.key2rank = {}
        self.keys = []
        self.best_latency = float(name2seq['latency'][0])
        # keys
        for rank, (key, latency) in enumerate(zip(zip(*[name2seq[name] for name in key_names]), name2seq['latency'])):
            self.keys.append(key)
            self.key2rank[key] = rank
            self.key2latency[key] = float(latency)


def pick_schedule(work2data, key_names, enroll_top_k=5, target_num=30, out_dir='./outs/analyze'):
    candidate_keys = []
    for work, data in work2data.items():
        for i in range(enroll_top_k):
            candidate_keys.append(data.keys[i])
            assert data.key2rank[data.keys[i]] < enroll_top_k
    candidate_keys = list(set(candidate_keys))
    print('num of candidates: ', len(candidate_keys))
    key2latency = defaultdict(list)
    for work, data in work2data.items():
        for key in candidate_keys:
            key2latency[key].append(float(data.key2latency[key]))
    key2mean = {}
    for key in key2latency:
        key2mean[key] = float(np.mean(key2latency[key]))
    key_latency = sorted(key2mean.items(), key=lambda v: v[1])
    names = key_names + ['mean_latency'] + list(work2data.keys())
    rows = []
    for key, latency in key_latency[:target_num]:
        row = []
        row.extend(key)
        row.append(latency)
        for data in work2data.values():
            row.append(data.key2latency[key])
        rows.append(row)
    # final row calculate degradation
    row = []
    for _ in key_names:
        row.append(None)
    row.append(None)  # mean latency
    for data in work2data.values():
        best_latency = data.best_latency
        cur_latency = min(data.key2latency[key] for key, _ in key_latency[:target_num])
        row.append(best_latency / cur_latency)
    rows.append(row)
    print('chosen ', target_num)
    with open(os.path.join(out_dir, 'pick.txt'), 'w') as f:
        f.write(str(tabulate(rows, headers=names, floatfmt='.3f', missingval='-')))
    print(tabulate(rows, headers=names, floatfmt='.3f', missingval='-'))


def main(resolve_dir='./outs/resolve', out_dir='./outs/analyze'):
    work2data = {}
    for entry in os.scandir(resolve_dir):
        assert isinstance(entry, os.DirEntry)
        if entry.is_file():
            continue
        report_path = os.path.join(entry.path, 'report.txt')
        if not os.path.isfile(report_path):
            continue
        with open(report_path, 'r') as f:
            data = {}
            key_names = []
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    for name in line.split():
                        data[name] = []
                        if name != 'latency':
                            key_names.append(name)
                elif idx == 1:
                    pass
                else:
                    row = line.split()
                    assert len(data) == len(row)
                    for name, item in zip(data.keys(), row):
                        data[name].append(item)
            work2data[entry.name] = Data(name2seq=data, key_names=key_names)
        os.makedirs(os.path.join(out_dir, entry.name), exist_ok=True)
        if not os.path.exists(os.path.join(out_dir, entry.name, 'report.txt')):
            shutil.copy(report_path, os.path.join(out_dir, entry.name, 'report.txt'))
    pick_schedule(work2data, key_names)


if __name__ == '__main__':
    main()
