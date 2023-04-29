// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

    struct BFloat16
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

        constexpr BFloat16(unsigned short bits, from_bits_t)
            : x(bits){};
        inline BFloat16(float value) : x(detail::round_to_nearest_even(value))
        {
        }
        inline operator float() const
        {
            return detail::f32_from_bits(x);
        }
    };

    /// Arithmetic

    inline BFloat16 operator+(const BFloat16 &a, const BFloat16 &b)
    {
        return static_cast<float>(a) + static_cast<float>(b);
    }

    inline BFloat16 operator-(const BFloat16 &a, const BFloat16 &b)
    {
        return static_cast<float>(a) - static_cast<float>(b);
    }

    inline BFloat16 operator*(const BFloat16 &a, const BFloat16 &b)
    {
        return static_cast<float>(a) * static_cast<float>(b);
    }

    inline BFloat16 operator/(const BFloat16 &a, const BFloat16 &b)
    {
        return static_cast<float>(a) / static_cast<float>(b);
    }

    inline BFloat16 operator-(const BFloat16 &a)
    {
        return -static_cast<float>(a);
    }

    inline BFloat16 &operator+=(BFloat16 &a, const BFloat16 &b)
    {
        a = a + b;
        return a;
    }

    inline BFloat16 &operator-=(BFloat16 &a, const BFloat16 &b)
    {
        a = a - b;
        return a;
    }

    inline BFloat16 &operator*=(BFloat16 &a, const BFloat16 &b)
    {
        a = a * b;
        return a;
    }

    inline BFloat16 &operator/=(BFloat16 &a, const BFloat16 &b)
    {
        a = a / b;
        return a;
    }

    inline BFloat16 &operator|(BFloat16 &a, const BFloat16 &b)
    {
        a.x = a.x | b.x;
        return a;
    }

    inline BFloat16 &operator^(BFloat16 &a, const BFloat16 &b)
    {
        a.x = a.x ^ b.x;
        return a;
    }

    inline BFloat16 &operator&(BFloat16 &a, const BFloat16 &b)
    {
        a.x = a.x & b.x;
        return a;
    }

    /// Arithmetic with floats

    inline float operator+(BFloat16 a, float b)
    {
        return static_cast<float>(a) + b;
    }
    inline float operator-(BFloat16 a, float b)
    {
        return static_cast<float>(a) - b;
    }
    inline float operator*(BFloat16 a, float b)
    {
        return static_cast<float>(a) * b;
    }
    inline float operator/(BFloat16 a, float b)
    {
        return static_cast<float>(a) / b;
    }

    inline float operator+(float a, BFloat16 b)
    {
        return a + static_cast<float>(b);
    }
    inline float operator-(float a, BFloat16 b)
    {
        return a - static_cast<float>(b);
    }
    inline float operator*(float a, BFloat16 b)
    {
        return a * static_cast<float>(b);
    }
    inline float operator/(float a, BFloat16 b)
    {
        return a / static_cast<float>(b);
    }

    inline float &operator+=(float &a, const BFloat16 &b)
    {
        return a += static_cast<float>(b);
    }
    inline float &operator-=(float &a, const BFloat16 &b)
    {
        return a -= static_cast<float>(b);
    }
    inline float &operator*=(float &a, const BFloat16 &b)
    {
        return a *= static_cast<float>(b);
    }
    inline float &operator/=(float &a, const BFloat16 &b)
    {
        return a /= static_cast<float>(b);
    }

    /// Arithmetic with doubles

    inline double operator+(BFloat16 a, double b)
    {
        return static_cast<double>(a) + b;
    }
    inline double operator-(BFloat16 a, double b)
    {
        return static_cast<double>(a) - b;
    }
    inline double operator*(BFloat16 a, double b)
    {
        return static_cast<double>(a) * b;
    }
    inline double operator/(BFloat16 a, double b)
    {
        return static_cast<double>(a) / b;
    }

    inline double operator+(double a, BFloat16 b)
    {
        return a + static_cast<double>(b);
    }
    inline double operator-(double a, BFloat16 b)
    {
        return a - static_cast<double>(b);
    }
    inline double operator*(double a, BFloat16 b)
    {
        return a * static_cast<double>(b);
    }
    inline double operator/(double a, BFloat16 b)
    {
        return a / static_cast<double>(b);
    }

    /// Arithmetic with ints

    inline BFloat16 operator+(BFloat16 a, int b)
    {
        return a + static_cast<BFloat16>(b);
    }
    inline BFloat16 operator-(BFloat16 a, int b)
    {
        return a - static_cast<BFloat16>(b);
    }
    inline BFloat16 operator*(BFloat16 a, int b)
    {
        return a * static_cast<BFloat16>(b);
    }
    inline BFloat16 operator/(BFloat16 a, int b)
    {
        return a / static_cast<BFloat16>(b);
    }

    inline BFloat16 operator+(int a, BFloat16 b)
    {
        return static_cast<BFloat16>(a) + b;
    }
    inline BFloat16 operator-(int a, BFloat16 b)
    {
        return static_cast<BFloat16>(a) - b;
    }
    inline BFloat16 operator*(int a, BFloat16 b)
    {
        return static_cast<BFloat16>(a) * b;
    }
    inline BFloat16 operator/(int a, BFloat16 b)
    {
        return static_cast<BFloat16>(a) / b;
    }

    //// Arithmetic with int64_t

    inline BFloat16 operator+(BFloat16 a, int64_t b)
    {
        return a + static_cast<BFloat16>(b);
    }
    inline BFloat16 operator-(BFloat16 a, int64_t b)
    {
        return a - static_cast<BFloat16>(b);
    }
    inline BFloat16 operator*(BFloat16 a, int64_t b)
    {
        return a * static_cast<BFloat16>(b);
    }
    inline BFloat16 operator/(BFloat16 a, int64_t b)
    {
        return a / static_cast<BFloat16>(b);
    }

    inline BFloat16 operator+(int64_t a, BFloat16 b)
    {
        return static_cast<BFloat16>(a) + b;
    }
    inline BFloat16 operator-(int64_t a, BFloat16 b)
    {
        return static_cast<BFloat16>(a) - b;
    }
    inline BFloat16 operator*(int64_t a, BFloat16 b)
    {
        return static_cast<BFloat16>(a) * b;
    }
    inline BFloat16 operator/(int64_t a, BFloat16 b)
    {
        return static_cast<BFloat16>(a) / b;
    }

    // Overloading < and > operators, because std::max and std::min use them.

    inline bool operator>(BFloat16 &lhs, BFloat16 &rhs)
    {
        return float(lhs) > float(rhs);
    }

    inline bool operator<(BFloat16 &lhs, BFloat16 &rhs)
    {
        return float(lhs) < float(rhs);
    }

}

namespace std
{
    using bfloat16::BFloat16;
    /// emulate bfloat16 math by float
    /// Used by vec256<bfloat16_t>::map
    inline BFloat16 acos(BFloat16 a)
    {
        return std::acos(float(a));
    }
    inline BFloat16 asin(BFloat16 a)
    {
        return std::asin(float(a));
    }
    inline BFloat16 atan(BFloat16 a)
    {
        return std::atan(float(a));
    }
    inline BFloat16 erf(BFloat16 a)
    {
        return std::erf(float(a));
    }
    inline BFloat16 erfc(BFloat16 a)
    {
        return std::erfc(float(a));
    }
    inline BFloat16 exp(BFloat16 a)
    {
        return std::exp(float(a));
    }
    inline BFloat16 expm1(BFloat16 a)
    {
        return std::expm1(float(a));
    }
    inline BFloat16 log(BFloat16 a)
    {
        return std::log(float(a));
    }
    inline BFloat16 log10(BFloat16 a)
    {
        return std::log10(float(a));
    }
    inline BFloat16 log1p(BFloat16 a)
    {
        return std::log1p(float(a));
    }
    inline BFloat16 log2(BFloat16 a)
    {
        return std::log2(float(a));
    }
    inline BFloat16 ceil(BFloat16 a)
    {
        return std::ceil(float(a));
    }
    inline BFloat16 cos(BFloat16 a)
    {
        return std::cos(float(a));
    }
    inline BFloat16 floor(BFloat16 a)
    {
        return std::floor(float(a));
    }
    inline BFloat16 nearbyint(BFloat16 a)
    {
        return std::nearbyint(float(a));
    }
    inline BFloat16 sin(BFloat16 a)
    {
        return std::sin(float(a));
    }
    inline BFloat16 tan(BFloat16 a)
    {
        return std::tan(float(a));
    }
    inline BFloat16 sinh(BFloat16 a)
    {
        return std::sinh(float(a));
    }
    inline BFloat16 cosh(BFloat16 a)
    {
        return std::cosh(float(a));
    }
    inline BFloat16 tanh(BFloat16 a)
    {
        return std::tanh(float(a));
    }
    inline BFloat16 trunc(BFloat16 a)
    {
        return std::trunc(float(a));
    }
    inline BFloat16 lgamma(BFloat16 a)
    {
        return std::lgamma(float(a));
    }
    inline BFloat16 sqrt(BFloat16 a)
    {
        return std::sqrt(float(a));
    }
    inline BFloat16 rsqrt(BFloat16 a)
    {
        return 1.0 / std::sqrt(float(a));
    }
    inline BFloat16 abs(BFloat16 a)
    {
        return std::abs(float(a));
    }

    inline BFloat16 round(BFloat16 a)
    {
        return std::round(float(a));
    }

    inline bool isinf(BFloat16 a)
    {
        return std::isinf(float(a));
    }

    inline BFloat16 pow(BFloat16 a, double b)
    {
        return std::pow(float(a), b);
    }

    inline BFloat16 pow(BFloat16 a, BFloat16 b)
    {
        return std::pow(float(a), float(b));
    }
    inline BFloat16 fmod(BFloat16 a, BFloat16 b)
    {
        return std::fmod(float(a), float(b));
    }

    inline BFloat16 fma(BFloat16 a, BFloat16 b, BFloat16 c)
    {
        return std::fma(float(a), float(b), float(c));
    }

    template <>
    class numeric_limits<BFloat16>
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

        static constexpr BFloat16 min()
        {
            return BFloat16(0x0080, BFloat16::from_bits());
        }
        static constexpr BFloat16 lowest()
        {
            return BFloat16(0xFF7F, BFloat16::from_bits());
        }
        static constexpr BFloat16 max()
        {
            return BFloat16(0x7F7F, BFloat16::from_bits());
        }
        static constexpr BFloat16 epsilon()
        {
            return BFloat16(0x3C00, BFloat16::from_bits());
        }
        static constexpr BFloat16 round_error()
        {
            return BFloat16(0x3F00, BFloat16::from_bits());
        }
        static constexpr BFloat16 infinity()
        {
            return BFloat16(0x7F80, BFloat16::from_bits());
        }
        static constexpr BFloat16 quiet_NaN()
        {
            return BFloat16(0x7FC0, BFloat16::from_bits());
        }
        static constexpr BFloat16 signaling_NaN()
        {
            return BFloat16(0x7F80, BFloat16::from_bits());
        }
        static constexpr BFloat16 denorm_min()
        {
            return BFloat16(0x0001, BFloat16::from_bits());
        }
    };
}

typedef bfloat16::BFloat16 bfloat16_t;
