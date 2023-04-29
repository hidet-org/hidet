#pragma once
#include <cuComplex.h>
#define HIDET_HOST_DEVICE __host__ __device__ __forceinline__

template <typename T>
struct Complex {
    T real, imag;
    Complex() = default;
    HIDET_HOST_DEVICE Complex(T real, T imag) : real(real), imag(imag) {}
};

template<typename T>
HIDET_HOST_DEVICE
Complex<T> operator-(Complex<T> a) {
    return {-a.real, -a.imag};
}

template<typename T>
HIDET_HOST_DEVICE
Complex<T> operator+(Complex<T> a, Complex<T> b) {
    return {a.real + b.real, a.imag + b.imag};
}

template<typename T>
HIDET_HOST_DEVICE Complex<T> operator-(Complex<T> a, Complex<T> b) {
    return {a.real - b.real, a.imag - b.imag};
}

template<typename T>
HIDET_HOST_DEVICE Complex<T> operator*(Complex<T> a, Complex<T> b) {
    return {a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real};
}

template<typename T>
HIDET_HOST_DEVICE Complex<T> operator/(Complex<T> a, Complex<T> b);

template<>
HIDET_HOST_DEVICE Complex<float> operator/(Complex<float> a, Complex<float> b) {
    auto ret = cuCdivf({a.real, a.imag}, {b.real, b.imag});
    return {ret.x, ret.y};
}

template<>
HIDET_HOST_DEVICE Complex<double> operator/(Complex<double> a, Complex<double> b) {
    auto ret = cuCdiv({a.real, a.imag}, {b.real, b.imag});
    return {ret.x, ret.y};
}

template<typename T>
HIDET_HOST_DEVICE T real(Complex<T> a) {
    return a.real;
}

template<typename T>
HIDET_HOST_DEVICE T imag(Complex<T> a) {
    return a.imag;
}

template<typename T>
HIDET_HOST_DEVICE Complex<T> conj(Complex<T> a) {
    return {a.real, -a.imag};
}

template<typename T>
HIDET_HOST_DEVICE T abs(Complex<T> a);

template<>
HIDET_HOST_DEVICE float abs(Complex<float> a) {
    return cuCabsf({a.real, a.imag});
}

template<>
HIDET_HOST_DEVICE double abs(Complex<double> a) {
    return cuCabs({a.real, a.imag});
}

template<typename T>
HIDET_HOST_DEVICE Complex<T> make_complex(T x, T y) {
    return {x, y};
}

typedef Complex<float> complex64_t;
typedef Complex<double> complex128_t;
