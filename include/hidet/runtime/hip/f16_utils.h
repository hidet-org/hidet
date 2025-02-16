#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

struct half4 {
    half x, y, z, w;

    __device__ half4() {}
    __device__ half4(half x, half y, half z, half w) : x(x), y(y), z(z), w(w) {}
};

struct half8 {
    half x, y, z, w, a, b, c, d;

    __device__ half8() {}
    __device__ half8(half x, half y, half z, half w, half a, half b, half c, half d)
        : x(x), y(y), z(z), w(w), a(a), b(b), c(c), d(d) {}
};
