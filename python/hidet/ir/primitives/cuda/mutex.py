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
from typing import Union
from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.primitives.cuda.ldst import load, store
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import script, i32, attr
    from hidet.lang.cuda import syncthreads_and, threadIdx, atomic_cas, syncthreads

    @script
    def cuda_acquire_lock(mutex_lock: ~i32):
        attr.func_kind = 'cuda_device'
        attr.func_name = 'cuda_acquire_lock'
        status: i32 = 1
        while syncthreads_and(status == 1):
            if threadIdx.x == 0:
                status = atomic_cas(mutex_lock, 0, 1)

    assert isinstance(cuda_acquire_lock, Function)
    register_primitive_function(cuda_acquire_lock.name, cuda_acquire_lock)

    @script
    def cuda_release_lock(mutex_lock: ~i32):
        attr.func_kind = 'cuda_device'
        attr.func_name = 'cuda_release_lock'
        syncthreads()
        if threadIdx.x == 0:
            atomic_cas(mutex_lock, 1, 0)

    assert isinstance(cuda_release_lock, Function)
    register_primitive_function(cuda_release_lock.name, cuda_release_lock)

    @script
    def cuda_acquire_seq_semaphore(semaphore: ~i32, expect_status: i32):
        attr.func_kind = 'cuda_device'
        attr.func_name = 'cuda_acquire_seq_semaphore'
        actual_status = load(semaphore, space='global', sync='acquire', scope='gpu')
        while syncthreads_and(actual_status != expect_status):
            if threadIdx.x == 0:
                actual_status = load(semaphore, space='global', sync='acquire', scope='gpu')

    assert isinstance(cuda_acquire_seq_semaphore, Function)
    register_primitive_function(cuda_acquire_seq_semaphore.name, cuda_acquire_seq_semaphore)

    @script
    def cuda_release_seq_semaphore(semaphore: ~i32, status: i32):
        attr.func_kind = 'cuda_device'
        attr.func_name = 'cuda_release_seq_semaphore'
        syncthreads()
        if threadIdx.x == 0:
            store(semaphore, status, space='global', sync='release', scope='gpu')

    assert isinstance(cuda_release_seq_semaphore, Function)
    register_primitive_function(cuda_release_seq_semaphore.name, cuda_release_seq_semaphore)


def acquire_lock(addr: Expr, scope: str = 'gpu'):
    """
    Acquire the lock at given address.

    This function assumes that the data at given address is an integer containing either 0 or 1, in which
    0 indices the lock is free to acquire and 1 indices the lock has been acquired by another thread block.

    This function will wait the value of the lock to be 0 for acquire and acquire the lock to change its
    value from 0 to 1. The compare and swap operators are conducted in one transaction.

    Parameters
    ----------
    addr: Expr
        The pointer to the lock with integer data type.

    scope: str
        The scope of the lock. Currently, we only support 'gpu' level of scope.
    """
    if scope != 'gpu':
        raise NotImplementedError()
    return call_primitive_func('cuda_acquire_lock', [addr])


def release_lock(addr: Expr, scope: str = 'gpu'):
    """
    Release the lock at given address.

    This function assumes that the data at given address is an integer containing either 0 or 1, in which
    0 indices the lock is free to acquire and 1 indices the lock has been acquired by another thread block.

    This function will change the value of the lock from 1 to 0 to allow other thread block to acquire the
    lock.

    Parameters
    ----------
    addr: Expr
        The pointer to the lock with integer data type.

    scope: str
        The scope of the lock. Currently, we only support 'gpu' level of scope.
    """
    if scope != 'gpu':
        raise NotImplementedError()
    return call_primitive_func('cuda_release_lock', [addr])


def acquire_seq_semaphore(addr: Expr, expect_status: Union[int, Expr]):
    return call_primitive_func('cuda_acquire_seq_semaphore', [addr, expect_status])


def release_seq_semaphore(addr: Expr, store_status: Union[int, Expr]):
    return call_primitive_func('cuda_release_seq_semaphore', [addr, store_status])
