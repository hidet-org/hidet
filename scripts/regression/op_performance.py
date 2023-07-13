import os
import json
import torch
import hidet
import argparse
from result_entry import ResultEntry, ResultGroup, load_regression_data

device_name = str(hidet.cuda.properties().name, 'UTF-8')


def matmul_regression() -> ResultGroup:
    regression_data = load_regression_data()
    result_group = ResultGroup(name='Matrix Multiply Regression', device_name=device_name)
    matmul_data = regression_data[device_name]['matmul_shapes']
    for shape, perf in matmul_data.items():
        for dtype, latency in perf.items():
            result_group.add_entry(ResultEntry(shape, dtype, latency))
    return result_group


def fmha_regression() -> ResultGroup:
    pass

def conv2d_regression() -> ResultGroup:
    pass

def reduce_regression() -> ResultGroup:
    pass


def op_performance_regression(report_file):
    result_groups = []
    result_groups.append(matmul_regression())
    result_groups.append(fmha_regression())
    result_groups.append(conv2d_regression())
    result_groups.append(reduce_regression())
    with open(report_file, 'w') as f:
        f.write("---------------- Operator Performance Regression -----------------\n")
        for result_group in result_groups:
            if result_group is not None:
                f.write(str(result_group))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Operator Performance Regression')
    parser.add_argument(
        '--report',
        type=str,
        default='./report_op_performance.txt',
        help='Specify report output path'
    )
    args = parser.parse_args()
    op_performance_regression(args.report)