# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###################################################################################################
# Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
##################################################################################################/


def shiftr(a, s):
    return a >> s if s > 0 else shiftl(a, -s)


def shiftl(a, s):
    return a << s if s > 0 else shiftr(a, -s)


def countr_zero(x: int):
    return (x & -x).bit_length() - 1


def bit_count(x: int):
    return bin(x).count("1")


# A generic Swizzle functor
# 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
#                               ^--^  Base is the number of least-sig bits to keep constant
#                  ^-^       ^-^      Bits is the number of bits in the mask
#                    ^---------^      Shift is the distance to shift the YYY mask
#                                       (pos shifts YYY to the right, neg shifts YYY to the left)
#
# e.g. Given
# 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
# the result is
# 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
#
class Swizzle:
    def __init__(self, bits, base, shift):
        assert bits >= 0
        assert base >= 0
        assert abs(shift) >= bits
        self.bits = bits
        self.base = base
        self.shift = shift
        bit_msk = (1 << bits) - 1
        self.yyy_msk = bit_msk << (base + max(0, shift))
        self.zzz_msk = bit_msk << (base - min(0, shift))

    # operator ()    (transform integer)
    def __call__(self, offset):
        return offset ^ shiftr(offset & self.yyy_msk, self.shift)

    # Size of the domain
    def size(self):
        return 1 << (self.bits + self.base + abs(self.shift))

    # Size of the codomain
    def cosize(self):
        return self.size()

    # print and str
    def __str__(self):
        return f"SW_{self.bits}_{self.base}_{self.shift}"

    # error msgs and representation
    def __repr__(self):
        return f"Swizzle({self.bits},{self.base},{self.shift})"

    def __eq__(self, other: "Swizzle"):
        return self.bits == other.bits and self.base == other.base and self.shift == other.shift


def make_swizzle(y: int, z: int) -> Swizzle:
    bz = bit_count(z)
    by = bit_count(y)
    assert bz == by
    tz_y = countr_zero(y)
    tz_z = countr_zero(z)
    m = min(tz_y, tz_z) % 32
    s = int(tz_y) - int(tz_z)
    return Swizzle(bz, m, s)
