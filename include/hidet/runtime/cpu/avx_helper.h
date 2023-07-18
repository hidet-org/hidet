#include <immintrin.h>

static inline __m256
as_v8_f32_u32(__m256i x)
{
    union {
        __m256i _xi; __m256 _xf;
    } val = { ._xi = x};

    return val._xf;
}

static inline __m256i
as_v8_u32_f32(__m256 x)
{
    union {
        __m256i _xi; __m256 _xf;
    } val = { ._xf = x};

    return val._xi;
}

/*
 * p(x) = c7*x^7 + c6*x^6 + c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
 *      = ((c6+c7*x)*x2 + (c4+c5*x))*x4 + ((c2+c3*x)*x2 + (c0+c1*x))
 */

#define POLY_EVAL_7(x, c0, c1, c2, c3, c4, c5, c6, c7) ({               \
            __typeof(x) x2 = x * x;                                     \
            __typeof(x) x4 = x2 * x2;                                   \
            __typeof(x) q = mul_add(mul_add(mul_add(c7, x, c6),         \
                                            x2,                         \
                                            mul_add(c5, x, c4)),        \
                                    x4,                                 \
                                    mul_add(mul_add(c3, x, c2),         \
                                            x2,                         \
                                            mul_add(c1, x, c0)));       \
            q;                                                          \
        })

#define mul_add(x, y, z)                                        \
        _Generic((x),                                           \
                 float  : _mm_fmadd_ss,                         \
                 double : _mm_fmadd_sd,                         \
                 __m128 : _mm_fmadd_ps,                         \
                 __m128d: _mm_fmadd_pd,                         \
                 __m256 : _mm256_fmadd_ps,                      \
                 __m256d: _mm256_fmadd_pd,                      \
                 __m512 : _mm512_fmadd_ps,                      \
                 __m512d: _mm512_fmadd_pd)((x), (y), (z))
