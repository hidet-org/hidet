from hidet.ir.type import tensor_type
from hidet.ir.expr import var
from hidet.ir.task import Task, Grid
from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.implement import implement, impl_context
from hidet.backend import build
from hidet.tos.tensor import randn, empty


def get_task(N=1024, M=1024, K=1024):
    A = tensor_input('A', 'float32', [N, K], 'global')
    B = tensor_input('B', 'float32', [K, M], 'global')
    C = compute('C', [N, M], lambda i, j: reduce([K], lambda k: A[i, k] * B[k, j], 'sum'), scope='global')

    task = Task('gemm', C, [A, B, C], Grid())
    return task


def main():
    N, M, K = 2, 2, 2
    task = get_task(N, M, K)
    ir_module = implement(task)

    # force hidet to use naive implementers
    module = build(ir_module, output_dir='./outs')

    A = randn([N, K], 'float32', device='cuda')
    B = randn([K, M], 'float32', device='cuda')
    C = empty([N, M], 'float32', device='cuda')
    module['gemm'](A, B, C)
    print(A)
    print(B)
    print(C)


if __name__ == '__main__':
    main()

"""
================
./outs/source.cu

#include <cassert>
#include <cstdio>
#include <cstdint>
extern "C" {

__device__ __forceinline__ void gemm_bt128x128_bsz256_s2x2_block(float A[4], float B[4], float out[4]) {
  float v;
  int32_t i1 = threadIdx.x;
  #pragma unroll
  for (int32_t o0 = 0; (o0 < 2); o0 = (o0 + 1)) {
    if ((o0 < 2) && (i1 < 2)) {
      v = 0.0;
      #pragma unroll
      for (int32_t k = 0; (k < 2); k = (k + 1)) {
        v = (v + (A[((o0 * 2) + k)] * B[((k * 2) + i1)]));
      } 
      out[((o0 * 2) + i1)] = v;
    } 
  } 
}

__global__ void __launch_bounds__(256, 1) gemm_grid(float A[4], float B[4], float C[4]) {
  // label: block_task-128x128-block_size-256
  int32_t n_block_idx = blockIdx.x;
  gemm_bt128x128_bsz256_s2x2_block(&A[0], &B[0], &C[0]);
}

__host__ void gemm(int32_t num_args, int32_t *arg_types, void* *args) {
  assert(((void)"expect 3 args", (num_args == 3)));
  assert(((void)"The 0 th arg should be TensorType(ScalarType(float32), [2, 2], global)", (arg_types[0] == 3)));
  assert(((void)"The 1 th arg should be TensorType(ScalarType(float32), [2, 2], global)", (arg_types[1] == 3)));
  assert(((void)"The 2 th arg should be TensorType(ScalarType(float32), [2, 2], global)", (arg_types[2] == 3)));
  gemm_grid<<<1,256>>>((float*)args[0], (float*)args[1], (float*)args[2]);
}

}

=============
./outs/lib.so
The generated dynamic library.

"""
