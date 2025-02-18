#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

typedef __nv_fp8_e4m3 float8_e4m3;

// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html

// Binary Arithmetic. Upcast to __half (fp16), perform operation in fp16, then downcast to fp8 (unsafe)
inline __host__ __device__ __nv_fp8_e4m3 operator+(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return __nv_fp8_e4m3(static_cast<__half>(a) + static_cast<__half>(b));
}
inline __host__ __device__ __nv_fp8_e4m3 operator-(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return __nv_fp8_e4m3(static_cast<__half>(a) - static_cast<__half>(b));
}
inline __host__ __device__ __nv_fp8_e4m3 operator*(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return __nv_fp8_e4m3(static_cast<__half>(a) * static_cast<__half>(b));
}
inline __host__ __device__ __nv_fp8_e4m3 operator/(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return __nv_fp8_e4m3(static_cast<__half>(a) / static_cast<__half>(b));
}
inline __host__ __device__ __nv_fp8_e4m3 operator-(const __nv_fp8_e4m3 &a) {
    return __nv_fp8_e4m3(-static_cast<__half>(a));
}

// Unary Arithmetic
inline __host__ __device__ __nv_fp8_e4m3 operator+(const __nv_fp8_e4m3 &a) {
    return a;
}
inline __host__ __device__ __nv_fp8_e4m3 operator+=(__nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    a = a + b;
    return a;
}
inline __host__ __device__ __nv_fp8_e4m3 operator-=(__nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    a = a - b;
    return a;
}
inline __host__ __device__ __nv_fp8_e4m3 operator*=(__nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    a = a * b;
    return a;
}
inline __host__ __device__ __nv_fp8_e4m3 operator/=(__nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    a = a / b;
    return a;
}

// Comparators
inline __host__ __device__ bool operator==(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return static_cast<__half>(a) == static_cast<__half>(b);
}
inline __host__ __device__ bool operator!=(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return static_cast<__half>(a) != static_cast<__half>(b);
}
inline __host__ __device__ bool operator>(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return static_cast<__half>(a) > static_cast<__half>(b);
}
inline __host__ __device__ bool operator<(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return static_cast<__half>(a) < static_cast<__half>(b);
}
inline __host__ __device__ bool operator>=(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return static_cast<__half>(a) >= static_cast<__half>(b);
}
inline __host__ __device__ bool operator<=(const __nv_fp8_e4m3 &a, const __nv_fp8_e4m3 &b) {
    return static_cast<__half>(a) <= static_cast<__half>(b);
}

// Increment/Decrement. uses the efficient underlying implementation of __half operator++ and __half operator--
inline __host__ __device__ __nv_fp8_e4m3 operator++(__nv_fp8_e4m3 &a) {
    __half tmp = static_cast<__half>(a);
    return __nv_fp8_e4m3(++tmp);
}
inline __host__ __device__ __nv_fp8_e4m3 operator--(__nv_fp8_e4m3 &a) {
    __half tmp = static_cast<__half>(a);
    return __nv_fp8_e4m3(--tmp);
}

// Unary mathematical functions. All functions cast to f16, except for tanh and erf which cast to f32.
inline __device__ __nv_fp8_e4m3 __abs(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(__habs(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 sin(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hsin(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 cos(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hcos(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 tanhf(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(tanhf(static_cast<float>(a)));
}
inline __device__ __nv_fp8_e4m3 exp(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hexp(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 exp2(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hexp2(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 erff(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(erff(static_cast<float>(a)));
}
inline __device__ __nv_fp8_e4m3 sqrt(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hsqrt(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 rsqrt(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hrsqrt(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 log(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hlog(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 rint(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hrint(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 ceil(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hceil(static_cast<__half>(a)));
}
inline __device__ __nv_fp8_e4m3 floor(const __nv_fp8_e4m3 a) {
    return __nv_fp8_e4m3(hfloor(static_cast<__half>(a)));
}

// Binary mathematical functions. All functions cast to f16, except for pow which casts to f32.
inline __device__ __nv_fp8_e4m3 __min(const __nv_fp8_e4m3 a, const __nv_fp8_e4m3 b) {
    return __nv_fp8_e4m3(__hmin(static_cast<__half>(a), static_cast<__half>(b)));
}
inline __device__ __nv_fp8_e4m3 __max(const __nv_fp8_e4m3 a, const __nv_fp8_e4m3 b) {
    return __nv_fp8_e4m3(__hmax(static_cast<__half>(a), static_cast<__half>(b)));
}
inline __device__ __nv_fp8_e4m3 powf(const __nv_fp8_e4m3 a, const __nv_fp8_e4m3 b) {
    return __nv_fp8_e4m3(powf(static_cast<float>(a), static_cast<float>(b)));
}

// Ternary mathematical functions
inline __device__ __nv_fp8_e4m3 __fma(const __nv_fp8_e4m3 a, const __nv_fp8_e4m3 b, const __nv_fp8_e4m3 c) {
    return __nv_fp8_e4m3(__hfma(static_cast<__half>(a), static_cast<__half>(b), static_cast<__half>(c)));
}
