import torch
import hidet
import argparse
from result_entry import ResultEntry, ResultGroup

# [M, N, K]
matmul_shapes = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16, 1024, 1024),
    (16, 4096, 4096),
    (16, 8192, 8192),
    (64, 1024, 1024),
    (64, 4096, 4096),
    (64, 8192, 8192),
    (1024, 64, 1024),
    (4096, 64, 4096),
    (8192, 64, 8192),
    (8192, 8192, 8176),
]
# [seqlen_q, seqlen_kv, hdim]
fmha_shapes = [
    (4096, 4096, 64),
    (4096, 4096, 128),
    (2048, 2048, 64),
    (2048, 2048, 128),
    (1024, 1024, 64),
    (1024, 1024, 128),
]
def batch_matmul_regression() -> ResultGroup:
    pass

def matmul_f16_regression() -> ResultGroup:
    pass

def fmha_regression() -> ResultGroup:
    pass

def conv2d_regression() -> ResultGroup:
    pass

def reduce_regression() -> ResultGroup:
    pass


def op_performance_regression(report_file):
    result_groups = []
    result_groups.append(batch_matmul_regression())
    result_groups.append(matmul_f16_regression())
    result_groups.append(fmha_regression())
    result_groups.append(conv2d_regression())
    result_groups.append(reduce_regression())
    with open(report_file, 'w') as f:
        f.write("ToDo: Operator Performance Regression")

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