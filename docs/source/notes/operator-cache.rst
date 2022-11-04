Operator Cache
==============

Hidet maintains a cache for the compiled operators. The default location of the cache directory depends on whether
the package is under a git repository or not.

- If it is under a git repository, the cache directory will be ``repo/.hidet_cache`` where ``repo`` is the root
  directory of the git repository.
- Otherwise, the cache directory will be ``~/.cache/hidet``.

You can you use :func:`hidet.option.get_cache_dir` to get the path to the cache directory.

>>> import hidet
>>> hidet.option.get_cache_dir()
/home/username/.cache/hidet

The hidet cache directory is organized as follows::

    cache_root
    |-- onnx                          (automatically downloaded ONNX models)
    |   |-- bert.onnx
    |   `-- resnet50.onnx
    `-- ops
        |-- cpu_space_0
        |-- cuda_space_0              (<target>_space_<space>)
        |   |-- add
        |   |   `-- 41920731adb3acf4  (task string hash)
        |   |       |-- lib.so        (compiled kernel)
        |   |       |-- nvcc_log.txt  (compilation command and nvcc output)
        |   |       |-- source.cu     (kernel source code)
        |   |       `-- task.txt      (task string)
        |   `-- matmul
        |       `-- 92dfdc1734b3854d
        |           |-- lib.so
        |           |-- nvcc_log.txt
        |           |-- source.cu
        |           `-- task.txt
        `-- cuda_space_2
            `-- batch_matmul
                |-- 2e4ce0f773c8d25c
                |   |-- lib.so
                |   |-- nvcc_log.txt
                |   |-- source.cu
                |   `-- summary.txt
                `-- 3b8442fa440916f7
                    |-- lib.so
                    |-- nvcc_log.txt
                    |-- source.cu
                    |-- summary.txt
                    `-- task.txt

When we run an operator, we first check if the operator is already compiled and cached. If so, we load the cached kernel
in ``lib.so`` and run it without recompilation. Otherwise, we compile the operator and cache the compiled kernel.

To change the cache directory, you can use :func:`hidet.option.cache_dir`:

>>> hidet.option.cache_dir('/path/to/cache')
