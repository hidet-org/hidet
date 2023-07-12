import torch
import hidet

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
    print(report_file)
    pass

if __name__ == '__main__':
    op_performance_regression('./report_op_performance.txt')