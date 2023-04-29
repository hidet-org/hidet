#include <complex>

typedef std::complex<float> complex64_t;
typedef std::complex<double> complex128_t;

float abs(const complex64_t &a) {
    return std::abs(a);
}

double abs(const complex128_t &a) {
    return std::abs(a);
}

float real(const complex64_t &a) {
    return std::real(a);
}

double real(const complex128_t &a) {
    return std::real(a);
}

float imag(const complex64_t &a) {
    return std::imag(a);
}

double imag(const complex128_t &a) {
    return std::imag(a);
}

complex64_t conj(const complex64_t &a) {
    return std::conj(a);
}

complex128_t conj(const complex128_t &a) {
    return std::conj(a);
}

complex64_t make_complex(float x, float y) {
    return complex64_t(x, y);
}

complex128_t make_complex(double x, double y) {
    return complex128_t(x, y);
}
