from typing import Tuple, Dict, List
import re
import os
import os.path


class Result:
    def __init__(self, commit=None, device=None, row_names=None, column_names=None, table=None):
        self.commit: str = commit
        self.device: str = device
        self.row_names: List[str] = row_names
        self.column_names: List[str] = column_names
        self.table: List[List[float]] = table

    def __str__(self):
        table = [['Workload'] + self.column_names]
        for idx, row in enumerate(self.table):
            items = [self.row_names[idx]] + [f'{v:.3f}' for v in row]
            table.append(items)
        column_width = [0] * (len(self.column_names) + 1)
        for i, row in enumerate(table):
            for j, item in enumerate(row):
                column_width[j] = max(column_width[j], len(item) + 1)
        format_strings = [f'{{:>{column_width[j]}}}' for j in range(len(table[0]))]
        lines = [
            '{:>10}: {}'.format('Commit', self.commit),
            '{:>10}: {}'.format('Device', self.device),
            *[
                ', '.join([format_strings[j].format(item) for j, item in enumerate(row)]) for i, row in enumerate(table)
            ],
            ""
        ]
        return "\n".join(lines)


def analyze(report_path) -> Result:
    order = [
        'Reference',
        'HidetNaive',
        'HidetNoPipe',
        'HidetNoPipeLdg',
        'HidetSoftPipe',
        'HidetSoftPipeLdg',
        'HidetSoftPipeLdgWb',
        'HidetSoftPipePred',
        'HidetMatmul',
        'Opt',
        'cutlas',
        'cuBLAS',
        'cuBLAS (TC)',
    ]
    with open(report_path, 'r') as f:
        # read block information
        blocks = [[]]
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                blocks.append([])
            else:
                blocks[-1].append(line)
        blocks = [block for block in blocks if len(block) > 0]

        # info block
        result = Result()
        info_string = "\n".join(blocks[0])
        result.commit = re.search(r'Repo commit.*\((.*)\)', info_string).group(1)
        result.device = re.search(r'((GPU = )|(GPU: ))(.*)', info_string).group(4)
        blocks = blocks[1:]
        # print(result.commit)
        # print(result.device)

        # benchmark result
        # (workload, variant) -> latency
        data: Dict[Tuple[str, str], float] = {}
        for block in blocks:
            workload = re.search(r':(.*)', block[0]).group(1).replace(' ', '')
            for line in block[1:]:
                variant, latency = re.search(r'(?P<variant>\w*): (?P<latency>.*) \(', line).group('variant', 'latency')
                data[(workload, variant)] = float(latency)
        # print(data)
        workloads = sorted(set([workload for workload, variant in data.keys()]), key=lambda k: tuple(int(v) for v in k.split('x')))
        variants = sorted(set([variant for workload, variant in data.keys()]), key=lambda v: order.index(v) if v in order else len(order) + hash(v))
        result.row_names = workloads
        result.column_names = variants
        result.table = []
        for workload in workloads:
            result.table.append([])
            for variant in variants:
                if (workload, variant) in data:
                    val = data[(workload, variant)]
                else:
                    val = 0.0
                result.table[-1].append(val)
    return result


def main():
    report_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'report')
    for name in os.listdir(report_dir):
        name = os.path.join(report_dir, name)
        if os.path.isfile(name) and name.endswith('.report'):
            result = analyze(name)
            with open(name.replace('.report', '.summary'), 'w') as f:
                f.write(str(result))


if __name__ == '__main__':
    main()
