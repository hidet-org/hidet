#pragma once
#include <cstdint>
#include <cstdio>

namespace hidet {

struct alignas(1) float8_e4m3 {
    uint8_t v;

    float8_e4m3() {}

    // define the conversion between float and float8_e4m3
    __host__ __device__ float8_e4m3(float f) {
        uint32_t bits = *reinterpret_cast<uint32_t *>(&f);
        uint32_t sign = (bits >> 31) & 0x1;
        uint32_t exponent = (bits >> 23) & 0xFF;
        uint32_t mantissa = bits & 0x7FFFFF;

        // special cases
        if (exponent == 0xFF) {
            // NaN or Inf
            v = sign << 7 | 0x7F;
            return;
        }

        if (exponent == 0) {
            // zero or denormalized case
            v = sign << 7;
            return;
        }

        if (exponent > uint32_t(120 + 0x1F)) {
            // overflow to NaN
            v = sign << 7 | 0x7F;
            return;
        }
        if (exponent <= uint32_t(120)) {
            // underflow to subnormal or zero
            v = sign << 7 | ((mantissa | 0x800000) >> (uint32_t(120 + 21) - exponent));
            return;
        }

        mantissa = mantissa >> 20;
        v = sign << 7 | (exponent - uint32_t(120)) << 3 | mantissa;
    }
    __host__ __device__ operator float() const {
        uint32_t sign = (v >> 7) & 0x1;
        uint32_t exponent = (v >> 3) & 0xF;
        uint32_t mantissa = v & 0x7;
        uint32_t bits;

        if (exponent == 0xF and mantissa == 0x7) {
            // NaN
            bits = sign << 31 | 0x7F888888;
        } else if (exponent) {
            // normalized case
            bits = sign << 31 | (exponent + uint32_t(120)) << 23 | mantissa << 20;
        } else {
            if (mantissa == 0) {
                // zero
                bits = sign << 31;
            } else {
                // subnormal
                exponent = uint32_t(121);
                while ((mantissa & 0x8) == 0) {
                    mantissa = mantissa << 1;
                    exponent -= 1;
                }
                mantissa = mantissa & 0x7;
                bits = sign << 31 | exponent << 23 | mantissa << 20;
            }
        }
        return *reinterpret_cast<float *>(&bits);
    }
};
}  // namespace hidet

typedef hidet::float8_e4m3 float8_e4m3;
