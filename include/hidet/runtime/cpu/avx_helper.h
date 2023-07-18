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

