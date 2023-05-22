#pragma once
#include <hidet/runtime/common.h>

#define TENSOR_MAGIC 0x18F9E2C3
#define MAX_NUM_NDIM 32

/*
Tensor Format on Disk
  int32     int32   int32  int32      int32      ...   int32           sizeof(dtype) * prod(shape)
| device |  dtype | ndim  |shape[0] | shape[1] | ... | shape[ndim-1] | data                        |
*/

struct Tensor {
    int device;
    int dtype;
    int ndim;
    int shape[MAX_NUM_NDIM];    // MAX_NUM_NDIM is only the limitation of this implementation, not the format
    void *data;
};

DLL hidet_save_tensor(const char *path, Tensor *tensor);

DLL Tensor* hidet_load_tensor(const char *path);
