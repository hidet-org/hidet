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
#include <hidet/runtime/int_fastdiv.h>

HOST_DEVICE void calculate_magic_numbers(int d, int &m, int &s, int &as) {
    if (d == 1) {
        m = 0;
        s = -1;
        as = 1;
        return;
    } else if (d == -1) {
        m = 0;
        s = -1;
        as = -1;
        return;
    }

    int p;
    unsigned int ad, anc, delta, q1, r1, q2, r2, t;
    const unsigned two31 = 0x80000000;
    ad = (d == 0) ? 1 : abs(d);
    t = two31 + ((unsigned int)d >> 31);
    anc = t - 1 - t % ad;
    p = 31;
    q1 = two31 / anc;
    r1 = two31 - q1 * anc;
    q2 = two31 / ad;
    r2 = two31 - q2 * ad;
    do {
        ++p;
        q1 = 2 * q1;
        r1 = 2 * r1;
        if (r1 >= anc) {
            ++q1;
            r1 -= anc;
        }
        q2 = 2 * q2;
        r2 = 2 * r2;
        if (r2 >= ad) {
            ++q2;
            r2 -= ad;
        }
        delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));
    m = q2 + 1;
    if (d < 0) m = -m;
    s = p - 32;

    if ((d > 0) && (m < 0))
        as = 1;
    else if ((d < 0) && (m > 0))
        as = -1;
    else
        as = 0;
}
