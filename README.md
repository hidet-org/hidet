# HIDET: Hierarchical Decomposable Task Scheduler

## Motivation
Highly optimized tensor programs are necessary to run deep learning models on various hardware efficiently. 
There are two kinds of state-of-the-art tensor programs: hand-crafted tensor programs by hardware vendors, and auto-generated tensor programs. 
The former ones (e.g., cuBLAS, and MKL) are hard to generalize to general workloads (e.g., the fused subgraph in a neural network model) while the later ones (e.g., Ansor, Halide-AutoScheduler) suffer from the limited optimization techniques (e.g., hard to implement software pipeline and use tensor core). 
To support a wider range of computation workloads while be able to implement wide range of hardware specific optimizations, we propose Hidet, a tensor program generator based on hierarchical decomposable tasks. 
The input of Hidet is task, that contains the computation definition of the workload, the input specifications (e.g., strides of tensors), and the worker (e.g., a grid, thread block, warp or thread in cuda), and the output is a highly optimized tensor program to fulfill the given task. 
Hidet allows the hardware experts to provide implementers that implement the given task by either decomposing it into a subtask or writing Hidet low-level program, which allows Hidet to support general workloads while be capable of implementing hardware-specific optimizations.


## Method
The core idea of Hidet is to represent each workload (e.g., convolution, matrix multiplication) into decomposable **task**. Each task contains three components: the computation definition of the target workload, the input specification (e.g., the strides and memory scope of tensor), and the worker that would work on the task (e.g., grid, thread block, warp and thread). Each task can derive subtasks. Given a task, we can define custom decompose rules to implement the task or directly implement it (e.g., for the workload assigned to a thread).

There are several advantages using this abstraction:
1. Given the rules, it is an end-to-end generation of tensor program. People do not need to define the template. Besides, more kinds of rules can be defined to implement the task comparing to Ansor (This is just a claim now, I need more experiments to back it up).
2. Each task is isolated, and can be benchmarked. Thus, we can profile its performance. For example, if we want to know whether a warp-task is good, we can write a dummy kernel to execute the same warp to get a proxy number of its performance. Each task's output is a function (though finally they will be inlined)
3. This abstraction is platform-agnostic, we can define the CPU workers (e.g., multi-cores, single-thread). But more investigation on effecition CPU (x86, arm) kernels is needed.
4. The primitive operators (e.g., wmma in cuda) can be described as primitive task and fit into the scheduler easily.

## Build
### Build the C++ Backend of Hidet
1. Create build directory (under the root directory of hidet)
   ```bash
   mkdir build
   ```
2. Build the C++ backend
   ```bash
   cd build; cmake -DCMAKE_BUILD_TYPE=Release ..; make -j4
   ```
   Once the building completes, you should be able to find a dynamic library with path `build/lib/libhidet.so`.
### Config Python Library of Hidet
1. Install dependencies (under the root directory of hidet)
   ```bash 
   pip install -r requirements.txt
   ```
2. Config `PYTHONPATH`. Please add the following commands to the end of your shell init script (e.g., `~/.bashrc` for Bash) and restart the shell.
   ```bash 
   export HIDET_HOME=/path/to/hidet
   export PYTHONPATH=$PYTHONPATH:$HIDET_HOME/python
   ```
3. Validate the build of C++ backend and config of python package by running the following command 
   ```bash
   python -c "import hidet"
   ```
   No error means successful installation.

## Examples
An example is in `examples/matmul` to show how to define a matrix multiplication kernel using the hidet IR.

## Usage
A demo is in this file `experiments/demo/main.py`.

```python
from hidet.ir.type import tensor_type
from hidet.ir.expr import var
from hidet.ir.task import Task, Grid
from hidet.ir.dialects.compute import tensor_input, reduce_sum, compute
from hidet.runtime.value import randn, empty
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaGridNaiveImplementer, CudaThreadNaiveImplementer
from hidet.backend import build


def get_task(N=1024, M=1024, K=1024):
    k = var('k')

    A = tensor_input('A', 'float32', [N, K])
    B = tensor_input('B', 'float32', [K, M])
    C = compute('C', [N, M], lambda i, j: reduce_sum(A[i, k] * B[k, j], axes=k, shape=[K]))

    params_type = [
        tensor_type('global', 'float32', [N, K], [K, 1]),
        tensor_type('global', 'float32', [K, M], [M, 1]),
        tensor_type('global', 'float32', [N, M], [M, 1])
    ]
    task = Task('gemm', C, [A, B, C], params_type, Grid())
    return task


def main():
    N, M, K = 2, 2, 2
    task = get_task(N, M, K)
    ir_module = implement(task)

    # force hidet to use naive implementers
    with impl_context(allowed=[CudaGridNaiveImplementer, CudaThreadNaiveImplementer]):
        module = build(ir_module, output_dir='./outs')

    A = randn([N, K], 'float32', 'global', seed=1)
    B = randn([K, M], 'float32', 'global', seed=3)
    C = empty([N, M], 'float32', 'global')
    module['gemm'](A, B, C)
    print(A)
    print(B)
    print(C)


if __name__ == '__main__':
    main()
```

Run the code with
```bash
cd experiments/demo; python main.py
```

You will get output of matrices A, B, and C.
```text
[[6. 3.]
 [2. 4.]]
[[2. 4.]
 [0. 1.]]
[[12. 27.]
 [ 4. 12.]]
```

The generated cuda c code:
```c
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
```

After executing `experiments/demo/main.py`, you will find a directory `outs` in your current working directory with
```text
info/
lib.so
source.cu
source.ptx
```
`source.cu` contains the cuda c source code. `source.ptx` contains the PTX code. And `lib.so` is the dynamic library contains the kernel.

Other examples can be found in `tests/unit_tests/test_matmul_correctness.py` and `experiments/demos/main.py`.


## Benchmark
You can benchmark the latency of different kernels running
```bash
cd experiments/benchmark; python main.py
```

After running, you will get a report as follows
```text
Repo commit 8027080 (802708080acf7c3b860cb1d9c688d004cd9baa50)
GPU = NVIDIA GeForce RTX 3070 Laptop GPU
Arch = Ampere
Compute Capacity = (8, 6)
Warmup/Number/Repeat = 5 / 1 / 10
Use brute-force resolver = True

Workload (N x M x K): 1024 x 1024 x 1024
           Reference: 2.202 (std 0.001) ms [2.202 2.202 2.203 2.201 2.200 2.202 2.204 2.201 2.203 2.201]
                 Opt: 0.261 (std 0.002) ms [0.261 0.264 0.260 0.259 0.259 0.260 0.260 0.260 0.262 0.259]
              cutlas: 0.271 (std 0.001) ms [0.272 0.271 0.271 0.270 0.271 0.272 0.271 0.271 0.271 0.271]
              cuBLAS: 0.244 (std 0.001) ms [0.244 0.245 0.243 0.244 0.244 0.244 0.244 0.244 0.245 0.243]
          HidetNaive: 2.188 (std 0.003) ms [2.194 2.191 2.187 2.185 2.189 2.187 2.186 2.187 2.184 2.188]
         HidetNoPipe: 0.310 (std 0.001) ms [0.310 0.312 0.309 0.308 0.309 0.309 0.311 0.308 0.309 0.308]
      HidetNoPipeLdg: 0.314 (std 0.002) ms [0.315 0.315 0.312 0.313 0.316 0.311 0.314 0.315 0.315 0.313]
       HidetSoftPipe: 0.301 (std 0.002) ms [0.307 0.301 0.301 0.299 0.301 0.300 0.301 0.299 0.300 0.299]
    HidetSoftPipeLdg: 0.280 (std 0.001) ms [0.281 0.283 0.279 0.282 0.282 0.279 0.280 0.281 0.279 0.282]

Workload (N x M x K): 2048 x 2304 x 768
           Reference: 7.759 (std 0.002) ms [7.760 7.758 7.757 7.760 7.757 7.760 7.762 7.758 7.756 7.759]
                 Opt: 0.791 (std 0.001) ms [0.791 0.791 0.791 0.791 0.790 0.791 0.792 0.792 0.792 0.790]
              cutlas: 0.838 (std 0.008) ms [0.842 0.842 0.842 0.843 0.843 0.842 0.843 0.842 0.826 0.819]
              cuBLAS: 0.745 (std 0.001) ms [0.745 0.744 0.745 0.745 0.747 0.746 0.744 0.745 0.747 0.743]
          HidetNaive: 7.461 (std 0.293) ms [6.620 7.318 7.612 7.610 7.613 7.611 7.595 7.542 7.543 7.543]
         HidetNoPipe: 0.916 (std 0.003) ms [0.914 0.916 0.921 0.914 0.914 0.912 0.914 0.913 0.919 0.920]
      HidetNoPipeLdg: 0.919 (std 0.004) ms [0.922 0.919 0.915 0.915 0.927 0.915 0.920 0.915 0.922 0.914]
       HidetSoftPipe: 0.863 (std 0.002) ms [0.865 0.865 0.862 0.864 0.863 0.865 0.857 0.863 0.862 0.861]
    HidetSoftPipeLdg: 0.808 (std 0.004) ms [0.810 0.811 0.808 0.799 0.808 0.817 0.805 0.811 0.808 0.807]
```

You can change the size of workloads you want to benchmark by updating the following lines at the beginning of `benchmark` function:
```python
workloads = [
    (1024, 1024, 1024),
    (2048, 2304, 768),
]
```

## Implementation

Hidet is a new tensor program generator/compiler and does not depend on existing tensor program generators. For fast prototype, we implement most components of Hidet in python, and only leave the necessary runtime support and baselines to C++. 

Because this project is under active development, there is no official documentation. Please contact with Yaoyao Ding (dingyaoyao.cs@gmail.com) if you have any question during reading the code. 

