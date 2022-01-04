import os
from hidet.core.compute import Axis, tensor_input, reduce_sum, compute
from hidet.core.task import Task
from hidet.core.worker import Grid
from hidet.ir.type import tensor_type
from hidet.ir.func import IRModule
from hidet.scheduler.naive import naive_scheduler_grid
from hidet.backend.cuda import build
from hidet.backend.cuda.transforms import split_host_device_pass, flatten_global_tensor
from hidet.runtime.module import CompiledModule
from hidet.runtime.value import TensorValue


def get_task(N=1024, M=1024, K=1024):
    k = Axis(K)

    A = tensor_input('A', 'float32', [N, K])
    B = tensor_input('B', 'float32', [K, M])
    C = compute('C', [N, M], lambda i, j: reduce_sum(A[i, k] * B[k, j], axis=k))

    params_type = [
        tensor_type('global', 'float32', [N, K], [K, 1]),
        tensor_type('global', 'float32', [K, M], [M, 1]),
        tensor_type('global', 'float32', [N, M], [M, 1])
    ]
    task = Task('gemm.grid', C, [A, B, C], params_type, Grid())
    return task


def lower(ir_module: IRModule) -> IRModule:
    ir_module = split_host_device_pass()(ir_module)
    ir_module = flatten_global_tensor()(ir_module)
    return ir_module


def build_module(ir_module: IRModule, out_dir) -> CompiledModule:
    os.makedirs(out_dir, exist_ok=True)
    module = build(ir_module, out_dir)
    return module


def main():
    N, M, K = 2, 2, 2
    task = get_task(N, M, K)
    ir_module = naive_scheduler_grid(task)
    ir_module = lower(ir_module)
    module = build_module(ir_module, out_dir='./outs')

    A = TensorValue.randn([N, K], 'float32', 'global', seed=1)
    B = TensorValue.randn([K, M], 'float32', 'global', seed=3)
    C = TensorValue.empty([N, M], 'float32', 'global')
    module['gemm.host'](A, B, C)
    print(A)
    print(B)
    print(C)


if __name__ == '__main__':
    main()

"""
================
./outs/source.cu

#include <cassert>
extern "C" {

__device__ __forceinline__ void gemm_grid_thread(float* A, int32_t v, float* B, int32_t v_1, float &v_2) {
  v_2 = 0.0;
  for (int32_t i = 0; (i < 2); i = (i + 1)) {
    v_2 = (v_2 + (A[((v * 2) + (i * 1))] * B[((i * 2) + (v_1 * 1))]));
  } 
}

__global__ void gemm_grid(float* A, float* B, float* C) {
  float out;
  int32_t iv = ((blockIdx.x * 256) + threadIdx.x);
  int32_t iv_1 = ((iv / 2) % 2);
  int32_t iv_2 = ((iv / 1) % 2);
  if (iv_1 < 2) {
    if (iv_2 < 2) {
      gemm_grid_thread(A, iv_1, B, iv_2, out);
      C[((iv_1 * 2) + (iv_2 * 1))] = out;
    } 
  } 
}

__host__ void gemm_host(int32_t num_args, int32_t* arg_types, void** args) {
  assert(((void)"expect 3 args", (num_args == 3)));
  assert(((void)"The 0 th arg should be TensorType(DataType(float32), [2, 2], global)", (arg_types[0] == 3)));
  assert(((void)"The 1 th arg should be TensorType(DataType(float32), [2, 2], global)", (arg_types[1] == 3)));
  assert(((void)"The 2 th arg should be TensorType(DataType(float32), [2, 2], global)", (arg_types[2] == 3)));
  gemm_grid<<<1,256>>>((float*)args[0], (float*)args[1], (float*)args[2]);
}

}

=============
./outs/lib.so
The generated dynamic library.

"""
