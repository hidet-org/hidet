# HIDET: Hierarchical Decomposable Task Scheduler

## Motivation
The core idea of Hidet is to represent each workload (e.g., convolution, matrix multiplication) into decomposable **task**. Each task contains three components: the computation definition of the target workload, the input specification (e.g., the strides and memory scope of tensor), and the worker that would work on the task (e.g., grid, thread block, warp and thread). Each task can derive subtasks. Given a task, we can define custom decompose rules to implement the task or directly implement it (e.g., for the workload assigned to a thread).

There are several advantages using this abstraction:
1. Given the rules, it is an end-to-end generation of tensor program. People do not need to define the template. Besides, more kinds of rules can be defined to implement the task comparing to Ansor (This is just a claim now, I need more experiments to back it up).
2. Each task is isolated, and can be benchmarked. Thus, we can profile its performance. For example, if we want to know whether a warp-task is good, we can write a dummy kernel to execute the same warp to get a proxy number of its performance. Each task's output is a function (though finally they will be inlined)
3. This abstraction is platform-agnostic, we can define the CPU workers (e.g., multi-cores, single-thread). But more investigation on effecition CPU (x86, arm) kernels is needed.


## Method
### Task
The definition of a task can be found in `python/hidet/core/task.py`.

### Scheduler
Currently, a naive task scheduler is implemented in `python/hidet/scheduler/naive.py`.

## Roadmap

-[x] The pipeline: define task, schedule, lower, codegen, build, runtime.
-[ ] Implement advanced decompose rules.

## Implementation
A new set of IR, lowering passes, and necessary runtime are implemented in python for fast prototype. 

