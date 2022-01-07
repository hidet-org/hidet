# HIDET: Hierarchical Decomposable Task Scheduler

## Motivation
The core idea of Hidet is to represent each workload (e.g., convolution, matrix multiplication) into decomposable **task**. Each task contains three components: the computation definition of the target workload, the input specification (e.g., the strides and memory scope of tensor), and the worker that would work on the task (e.g., grid, thread block, warp and thread). Each task can derive subtasks. Given a task, we can define custom decompose rules to implement the task or directly implement it (e.g., for the workload assigned to a thread).

There are several advantages using this abstraction:
1. Given the rules, it is an end-to-end generation of tensor program. People do not need to define the template. Besides, more kinds of rules can be defined to implement the task comparing to Ansor (This is just a claim now, I need more experiments to back it up).
2. Each task is isolated, and can be benchmarked. Thus, we can profile its performance. For example, if we want to know whether a warp-task is good, we can write a dummy kernel to execute the same warp to get a proxy number of its performance. Each task's output is a function (though finally they will be inlined)
3. This abstraction is platform-agnostic, we can define the CPU workers (e.g., multi-cores, single-thread). But more investigation on effecition CPU (x86, arm) kernels is needed.
4. The primitive operators (e.g., wmma in cuda) can be described as primitive task and fit into the scheduler easily.


## Method
### Task
The definition of a task can be found in `python/hidet/core/task.py`.

### Scheduler
Currently, a naive task scheduler is implemented in `python/hidet/scheduler/naive.py`.

## Roadmap

- [x] The pipeline: define task, schedule, lower, codegen, build, runtime.
- [ ] Implement advanced decompose rules.

## Implementation
A new set of IR, lowering passes, and necessary runtime are implemented in python for fast prototype.

## Usage
A demo is here `experiments/demo/main.py`

```python
import os
from hidet.ir.type import tensor_type
from hidet.ir.func import IRModule
from hidet.ir.task import Task, Grid
from hidet.ir.dialects.compute import Axis, tensor_input, reduce_sum, compute
from hidet.transforms import const_expr_simplifier
from hidet.backend.cuda import build
from hidet.backend.cuda.transforms import split_host_device_pass, flatten_global_tensor
from hidet.runtime.module import CompiledModule
from hidet.runtime.value import TensorValue
from hidet.implement import implement


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
    ir_module = const_expr_simplifier()(ir_module)
    return ir_module


def build_module(ir_module: IRModule, out_dir) -> CompiledModule:
    os.makedirs(out_dir, exist_ok=True)
    module = build(ir_module, out_dir)
    return module


def main():
    N, M, K = 2, 2, 2
    task = get_task(N, M, K)
    ir_module = implement(task)
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
Output:

[[6. 3.]
 [2. 4.]]
[[2. 4.]
 [0. 1.]]
[[12. 27.]
 [ 4. 12.]]
"""
```
Generated CUDA code:
```c++
#include <cassert>
extern "C" {

__device__ __forceinline__ void gemm_grid_thread(float* A, int32_t v, float* B, int32_t v_1, float &v_2) {
  v_2 = 0.0;
  for (int32_t i = 0; (i < 2); i = (i + 1)) {
    v_2 = (v_2 + (A[((v * 2) + i)] * B[((i * 2) + v_1)]));
  } 
}

__global__ void gemm_grid(float* A, float* B, float* C) {
  float out;
  int32_t iv = ((blockIdx.x * 256) + threadIdx.x);
  int32_t iv_1 = ((iv / 2) % 2);
  int32_t iv_2 = (iv % 2);
  if (iv_1 < 2) {
    if (iv_2 < 2) {
      gemm_grid_thread(A, iv_1, B, iv_2, out);
      C[((iv_1 * 2) + iv_2)] = out;
    } 
  } 
}

__host__ void gemm_host(int32_t num_args, int32_t* arg_types, void** args) {
  assert(((void)"expect 3 args", (num_args == 3)));
  assert(((void)"The 0 th arg should be TensorType(ScalarType(float32), [2, 2], global)", (arg_types[0] == 3)));
  assert(((void)"The 1 th arg should be TensorType(ScalarType(float32), [2, 2], global)", (arg_types[1] == 3)));
  assert(((void)"The 2 th arg should be TensorType(ScalarType(float32), [2, 2], global)", (arg_types[2] == 3)));
  gemm_grid<<<1,256>>>((float*)args[0], (float*)args[1], (float*)args[2]);
}

}
```

