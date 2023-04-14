/**
 * From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>
#include <cstring>
#include <stdint.h>
#include <limits>


namespace bfloat16
{
    namespace detail
    {
        inline float f32_from_bits(uint16_t src)
        {
            float res = 0;
            uint32_t tmp = src;
            tmp <<= 16;

            std::memcpy(&res, &tmp, sizeof(tmp));

            return res;
        }

        inline uint16_t bits_from_f32(float src)
        {
            uint32_t res = 0;

            std::memcpy(&res, &src, sizeof(res));

            return res >> 16;
        }

        inline uint16_t round_to_nearest_even(float src)
        {

            if (std::isnan(src))
            {
                return UINT16_C(0x7FC0);
            }
            else
            {
                union
                {
                    uint32_t U32;
                    float F32;
                };

                F32 = src;
                uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
                return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
            }
        }
    } // namespace detail

    struct bfloat16_t
    {
        uint16_t x;

        // bfloat16_t(const bfloat16_t&) = default;

        struct from_bits_t
        {
        };
        static constexpr from_bits_t from_bits()
        {
            return from_bits_t();
        }

        constexpr bfloat16_t(unsigned short bits, from_bits_t)
            : x(bits){};
        inline bfloat16_t(float value) : x(detail::round_to_nearest_even(value))
        {
        }
        inline operator float() const
        {
            return detail::f32_from_bits(x);
        }
    };

    /// Arithmetic

    inline bfloat16_t operator+(const bfloat16_t &a, const bfloat16_t &b)
    {
        return static_cast<float>(a) + static_cast<float>(b);
    }

    inline bfloat16_t operator-(const bfloat16_t &a, const bfloat16_t &b)
    {
        return static_cast<float>(a) - static_cast<float>(b);
    }

    inline bfloat16_t operator*(const bfloat16_t &a, const bfloat16_t &b)
    {
        return static_cast<float>(a) * static_cast<float>(b);
    }

    inline bfloat16_t operator/(const bfloat16_t &a, const bfloat16_t &b)
    {
        return static_cast<float>(a) / static_cast<float>(b);
    }

    inline bfloat16_t operator-(const bfloat16_t &a)
    {
        return -static_cast<float>(a);
    }

    inline bfloat16_t &operator+=(bfloat16_t &a, const bfloat16_t &b)
    {
        a = a + b;
        return a;
    }

    inline bfloat16_t &operator-=(bfloat16_t &a, const bfloat16_t &b)
    {
        a = a - b;
        return a;
    }

    inline bfloat16_t &operator*=(bfloat16_t &a, const bfloat16_t &b)
    {
        a = a * b;
        return a;
    }

    inline bfloat16_t &operator/=(bfloat16_t &a, const bfloat16_t &b)
    {
        a = a / b;
        return a;
    }

    inline bfloat16_t &operator|(bfloat16_t &a, const bfloat16_t &b)
    {
        a.x = a.x | b.x;
        return a;
    }

    inline bfloat16_t &operator^(bfloat16_t &a, const bfloat16_t &b)
    {
        a.x = a.x ^ b.x;
        return a;
    }

    inline bfloat16_t &operator&(bfloat16_t &a, const bfloat16_t &b)
    {
        a.x = a.x & b.x;
        return a;
    }

    /// Arithmetic with floats

    inline float operator+(bfloat16_t a, float b)
    {
        return static_cast<float>(a) + b;
    }
    inline float operator-(bfloat16_t a, float b)
    {
        return static_cast<float>(a) - b;
    }
    inline float operator*(bfloat16_t a, float b)
    {
        return static_cast<float>(a) * b;
    }
    inline float operator/(bfloat16_t a, float b)
    {
        return static_cast<float>(a) / b;
    }

    inline float operator+(float a, bfloat16_t b)
    {
        return a + static_cast<float>(b);
    }
    inline float operator-(float a, bfloat16_t b)
    {
        return a - static_cast<float>(b);
    }
    inline float operator*(float a, bfloat16_t b)
    {
        return a * static_cast<float>(b);
    }
    inline float operator/(float a, bfloat16_t b)
    {
        return a / static_cast<float>(b);
    }

    inline float &operator+=(float &a, const bfloat16_t &b)
    {
        return a += static_cast<float>(b);
    }
    inline float &operator-=(float &a, const bfloat16_t &b)
    {
        return a -= static_cast<float>(b);
    }
    inline float &operator*=(float &a, const bfloat16_t &b)
    {
        return a *= static_cast<float>(b);
    }
    inline float &operator/=(float &a, const bfloat16_t &b)
    {
        return a /= static_cast<float>(b);
    }

    /// Arithmetic with doubles

    inline double operator+(bfloat16_t a, double b)
    {
        return static_cast<double>(a) + b;
    }
    inline double operator-(bfloat16_t a, double b)
    {
        return static_cast<double>(a) - b;
    }
    inline double operator*(bfloat16_t a, double b)
    {
        return static_cast<double>(a) * b;
    }
    inline double operator/(bfloat16_t a, double b)
    {
        return static_cast<double>(a) / b;
    }

    inline double operator+(double a, bfloat16_t b)
    {
        return a + static_cast<double>(b);
    }
    inline double operator-(double a, bfloat16_t b)
    {
        return a - static_cast<double>(b);
    }
    inline double operator*(double a, bfloat16_t b)
    {
        return a * static_cast<double>(b);
    }
    inline double operator/(double a, bfloat16_t b)
    {
        return a / static_cast<double>(b);
    }

    /// Arithmetic with ints

    inline bfloat16_t operator+(bfloat16_t a, int b)
    {
        return a + static_cast<bfloat16_t>(b);
    }
    inline bfloat16_t operator-(bfloat16_t a, int b)
    {
        return a - static_cast<bfloat16_t>(b);
    }
    inline bfloat16_t operator*(bfloat16_t a, int b)
    {
        return a * static_cast<bfloat16_t>(b);
    }
    inline bfloat16_t operator/(bfloat16_t a, int b)
    {
        return a / static_cast<bfloat16_t>(b);
    }

    inline bfloat16_t operator+(int a, bfloat16_t b)
    {
        return static_cast<bfloat16_t>(a) + b;
    }
    inline bfloat16_t operator-(int a, bfloat16_t b)
    {
        return static_cast<bfloat16_t>(a) - b;
    }
    inline bfloat16_t operator*(int a, bfloat16_t b)
    {
        return static_cast<bfloat16_t>(a) * b;
    }
    inline bfloat16_t operator/(int a, bfloat16_t b)
    {
        return static_cast<bfloat16_t>(a) / b;
    }

    //// Arithmetic with int64_t

    inline bfloat16_t operator+(bfloat16_t a, int64_t b)
    {
        return a + static_cast<bfloat16_t>(b);
    }
    inline bfloat16_t operator-(bfloat16_t a, int64_t b)
    {
        return a - static_cast<bfloat16_t>(b);
    }
    inline bfloat16_t operator*(bfloat16_t a, int64_t b)
    {
        return a * static_cast<bfloat16_t>(b);
    }
    inline bfloat16_t operator/(bfloat16_t a, int64_t b)
    {
        return a / static_cast<bfloat16_t>(b);
    }

    inline bfloat16_t operator+(int64_t a, bfloat16_t b)
    {
        return static_cast<bfloat16_t>(a) + b;
    }
    inline bfloat16_t operator-(int64_t a, bfloat16_t b)
    {
        return static_cast<bfloat16_t>(a) - b;
    }
    inline bfloat16_t operator*(int64_t a, bfloat16_t b)
    {
        return static_cast<bfloat16_t>(a) * b;
    }
    inline bfloat16_t operator/(int64_t a, bfloat16_t b)
    {
        return static_cast<bfloat16_t>(a) / b;
    }

    // Overloading < and > operators, because std::max and std::min use them.

    inline bool operator>(bfloat16_t &lhs, bfloat16_t &rhs)
    {
        return float(lhs) > float(rhs);
    }

    inline bool operator<(bfloat16_t &lhs, bfloat16_t &rhs)
    {
        return float(lhs) < float(rhs);
    }

    /*
      The following function is inspired from the implementation in `musl`
      Link to License: https://git.musl-libc.org/cgit/musl/tree/COPYRIGHT
      ----------------------------------------------------------------------
      Copyright © 2005-2020 Rich Felker, et al.
      Permission is hereby granted, free of charge, to any person obtaining
      a copy of this software and associated documentation files (the
      "Software"), to deal in the Software without restriction, including
      without limitation the rights to use, copy, modify, merge, publish,
      distribute, sublicense, and/or sell copies of the Software, and to
      permit persons to whom the Software is furnished to do so, subject to
      the following conditions:
      The above copyright notice and this permission notice shall be
      included in all copies or substantial portions of the Software.
      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
      EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
      MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
      IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
      CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
      TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
      SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
      ----------------------------------------------------------------------
     */
    inline bfloat16_t nextafter(
        bfloat16_t from,
        bfloat16_t to)
    {
        // Reference:
        // https://git.musl-libc.org/cgit/musl/tree/src/math/nextafter.c
        using int_repr_t = uint16_t;
        using float_t = bfloat16_t;
        constexpr uint8_t bits = 16;
        union
        {
            float_t f;
            int_repr_t i;
        } ufrom = {from}, uto = {to};

        // get a mask to get the sign bit i.e. MSB
        int_repr_t sign_mask = int_repr_t{1} << (bits - 1);

        // short-circuit: if either is NaN, return NaN
        if (from != from || to != to)
        {
            return from + to;
        }

        // short-circuit: if they are exactly the same.
        if (ufrom.i == uto.i)
        {
            return from;
        }

        // mask the sign-bit to zero i.e. positive
        // equivalent to abs(x)
        int_repr_t abs_from = ufrom.i & ~sign_mask;
        int_repr_t abs_to = uto.i & ~sign_mask;
        if (abs_from == 0)
        {
            // if both are zero but with different sign,
            // preserve the sign of `to`.
            if (abs_to == 0)
            {
                return to;
            }
            // smallest subnormal with sign of `to`.
            ufrom.i = (uto.i & sign_mask) | int_repr_t{1};
            return ufrom.f;
        }

        // if abs(from) > abs(to) or sign(from) != sign(to)
        if (abs_from > abs_to || ((ufrom.i ^ uto.i) & sign_mask))
        {
            ufrom.i--;
        }
        else
        {
            ufrom.i++;
        }

        return ufrom.f;
    }
}

namespace std
{
    using bfloat16::bfloat16_t;
    /// emulate bfloat16 math by float
    /// Used by vec256<bfloat16_t>::map
    inline bfloat16_t acos(bfloat16_t a)
    {
        return std::acos(float(a));
    }
    inline bfloat16_t asin(bfloat16_t a)
    {
        return std::asin(float(a));
    }
    inline bfloat16_t atan(bfloat16_t a)
    {
        return std::atan(float(a));
    }
    inline bfloat16_t erf(bfloat16_t a)
    {
        return std::erf(float(a));
    }
    inline bfloat16_t erfc(bfloat16_t a)
    {
        return std::erfc(float(a));
    }
    inline bfloat16_t exp(bfloat16_t a)
    {
        return std::exp(float(a));
    }
    inline bfloat16_t expm1(bfloat16_t a)
    {
        return std::expm1(float(a));
    }
    inline bfloat16_t log(bfloat16_t a)
    {
        return std::log(float(a));
    }
    inline bfloat16_t log10(bfloat16_t a)
    {
        return std::log10(float(a));
    }
    inline bfloat16_t log1p(bfloat16_t a)
    {
        return std::log1p(float(a));
    }
    inline bfloat16_t log2(bfloat16_t a)
    {
        return std::log2(float(a));
    }
    inline bfloat16_t ceil(bfloat16_t a)
    {
        return std::ceil(float(a));
    }
    inline bfloat16_t cos(bfloat16_t a)
    {
        return std::cos(float(a));
    }
    inline bfloat16_t floor(bfloat16_t a)
    {
        return std::floor(float(a));
    }
    inline bfloat16_t nearbyint(bfloat16_t a)
    {
        return std::nearbyint(float(a));
    }
    inline bfloat16_t sin(bfloat16_t a)
    {
        return std::sin(float(a));
    }
    inline bfloat16_t tan(bfloat16_t a)
    {
        return std::tan(float(a));
    }
    inline bfloat16_t sinh(bfloat16_t a)
    {
        return std::sinh(float(a));
    }
    inline bfloat16_t cosh(bfloat16_t a)
    {
        return std::cosh(float(a));
    }
    inline bfloat16_t tanh(bfloat16_t a)
    {
        return std::tanh(float(a));
    }
    inline bfloat16_t trunc(bfloat16_t a)
    {
        return std::trunc(float(a));
    }
    inline bfloat16_t lgamma(bfloat16_t a)
    {
        return std::lgamma(float(a));
    }
    inline bfloat16_t sqrt(bfloat16_t a)
    {
        return std::sqrt(float(a));
    }
    inline bfloat16_t rsqrt(bfloat16_t a)
    {
        return 1.0 / std::sqrt(float(a));
    }
    inline bfloat16_t abs(bfloat16_t a)
    {
        return std::abs(float(a));
    }

    inline bfloat16_t pow(bfloat16_t a, double b)
    {
        return std::pow(float(a), b);
    }

    inline bfloat16_t pow(bfloat16_t a, bfloat16_t b)
    {
        return std::pow(float(a), float(b));
    }
    inline bfloat16_t fmod(bfloat16_t a, bfloat16_t b)
    {
        return std::fmod(float(a), float(b));
    }

    template <>
    class numeric_limits<bfloat16_t>
    {
    public:
        static constexpr bool is_signed = true;
        static constexpr bool is_specialized = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
        static constexpr auto has_denorm_loss =
            numeric_limits<float>::has_denorm_loss;
        static constexpr auto round_style = numeric_limits<float>::round_style;
        static constexpr bool is_iec559 = false;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 8;
        static constexpr int digits10 = 2;
        static constexpr int max_digits10 = 4;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -125;
        static constexpr int min_exponent10 = -37;
        static constexpr int max_exponent = 128;
        static constexpr int max_exponent10 = 38;
        static constexpr auto traps = numeric_limits<float>::traps;
        static constexpr auto tinyness_before =
            numeric_limits<float>::tinyness_before;

        static constexpr bfloat16_t min()
        {
            return bfloat16_t(0x0080, bfloat16_t::from_bits());
        }
        static constexpr bfloat16_t lowest()
        {
            return bfloat16_t(0xFF7F, bfloat16_t::from_bits());
        }
        static constexpr bfloat16_t max()
        {
            return bfloat16_t(0x7F7F, bfloat16_t::from_bits());
        }
        static constexpr bfloat16_t epsilon()
        {
            return bfloat16_t(0x3C00, bfloat16_t::from_bits());
        }
        static constexpr bfloat16_t round_error()
        {
            return bfloat16_t(0x3F00, bfloat16_t::from_bits());
        }
        static constexpr bfloat16_t infinity()
        {
            return bfloat16_t(0x7F80, bfloat16_t::from_bits());
        }
        static constexpr bfloat16_t quiet_NaN()
        {
            return bfloat16_t(0x7FC0, bfloat16_t::from_bits());
        }
        static constexpr bfloat16_t signaling_NaN()
        {
            return bfloat16_t(0x7F80, bfloat16_t::from_bits());
        }
        static constexpr bfloat16_t denorm_min()
        {
            return bfloat16_t(0x0001, bfloat16_t::from_bits());
        }
    };
}
