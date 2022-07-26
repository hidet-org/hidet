# Set the compute capacity of target GPU.
# A lower number is compatible with newer GPUs. Compute capacity of typical NVIDIA GPUs:
#   - 60: P100
#   - 70: V100
#   - 80: A100
#   - 61: RTX 10 Series
#   - 75: RTX 20 Series
#   - 86: RTX 30 Series
set(HIDET_CUDA_ARCH 60)

# Set build type
#  - Debug
#  - Release
set(HIDET_BUILD_TYPE Release)

# Whether to build TVM.
# Hidet does not depends on TVM, but we can build it to do benchmarking and comparison.
#  - ON: Build TVM
#  - OFF: Do not build TVM
set(HIDET_BUILD_TVM ON)

# Whether to build cudnn baseline
#  - ON: Build cudnn baseline, and search it automatically.
#  - A-Path-to-cuDNN: Use the given path to search cudnn library. Can be used to specify
#                     different version of cuDNN.
#  - OFF: Do not build cudnn baseline.
set(HIDET_BASELINE_CUDNN ON)
