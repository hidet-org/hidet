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
#include <complex>

typedef std::complex<float> complex64_t;
typedef std::complex<double> complex128_t;

static float abs(const complex64_t &a) {
    return std::abs(a);
}

static double abs(const complex128_t &a) {
    return std::abs(a);
}

static float real(const complex64_t &a) {
    return std::real(a);
}

static double real(const complex128_t &a) {
    return std::real(a);
}

static float imag(const complex64_t &a) {
    return std::imag(a);
}

static double imag(const complex128_t &a) {
    return std::imag(a);
}

static complex64_t conj(const complex64_t &a) {
    return std::conj(a);
}

static complex128_t conj(const complex128_t &a) {
    return std::conj(a);
}

static complex64_t make_complex(float x, float y) {
    return complex64_t(x, y);
}

static complex128_t make_complex(double x, double y) {
    return complex128_t(x, y);
}
