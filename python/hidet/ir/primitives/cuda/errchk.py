from hidet.ir.stmt import BlackBoxStmt


def check_cuda_error():
    stmt = BlackBoxStmt(
        r'''{cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) LOG(ERROR) << "CUDA error: " << '''
        r'''cudaGetErrorString(err) << "\n";}'''
    )
    return stmt
