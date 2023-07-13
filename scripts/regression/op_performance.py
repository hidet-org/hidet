import torch
import hidet
import argparse

# [M, N, K] where C_mxn = A_mxk @ B_kxn
matmul_shapes = [
    []
]
def batch_matmul_regression():

    pass

def matmul_f16_regression():
    pass

def fmha_regression():
    pass

def conv2d_regression():
    pass

def reduce_regression():
    pass


def op_performance_regression(report_file):
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