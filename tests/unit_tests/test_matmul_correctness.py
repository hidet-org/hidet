import numpy as np
import pytest

from hidet.backend import build
from hidet.baselines.matmul import matmul_opt
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaBlockStaticMatmulNoPipeImplementer, CudaBlockNaiveImplementer
from hidet.implement.cuda import CudaGridSplitImplementer, CudaGridNaiveImplementer, CudaWarpTransfer2dImplementer, CudaWarpFillValueImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer
from hidet.implement.cuda import CudaThreadNaiveImplementer
from hidet.implement.resolve import random_resolve
from hidet.ir.task import Grid, Host
from hidet.runtime.value import TensorValue, randn, scalar, zeros, full
from hidet.tasks.nn import matmul


@pytest.mark.parametrize('N,M,K,name,packed_func', [
    [256, 256, 256, 'opt', matmul_opt()]
])
def test_baseline(N, M, K, name, packed_func):
    task = matmul(N, M, K)
    A = randn([N, K], 'float32', 'host', seed=1)
    B = randn([K, M], 'float32', 'host', seed=3)
    C = zeros([N, M], 'float32', 'host')

    task.worker = Host()
    host_module = build(random_resolve(implement(task)), f'./outs/verify/host/{name}')

    GA, GB, GC = A.to_cuda(), B.to_cuda(), C.to_cuda()
    packed_func(scalar(N), scalar(M), scalar(K), GA, GB, GC)

    HA, HB, HC = A.to_cpu(), B.to_cpu(), C.to_cpu()
    host_module['matmul'](HA, HB, HC)
    np.testing.assert_allclose(GC.to_numpy(), HC.to_numpy())


@pytest.mark.parametrize('N,M,K,name,implementers', [
    (256, 256, 256, 'HidetNaive', (CudaGridNaiveImplementer, CudaThreadNaiveImplementer)),
    (256, 256, 256, 'HidetNoPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpFillValueImplementer)),
    (256, 256, 256, 'HidetSoftPipeLdgWb', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpFillValueImplementer)),
])
def test_hidet_variant(N, M, K, name, implementers):
    task = matmul(N, M, K)
    A = randn([N, K], 'float32', 'host', seed=1)
    B = randn([K, M], 'float32', 'host', seed=3)
    C = zeros([N, M], 'float32', 'host')

    task.worker = Grid()
    with impl_context(allowed=implementers):
        ir_module = implement(task)
        grid_module = build(random_resolve(ir_module, seed=1), f'./outs/verify/{name}')

    task.worker = Host()
    host_module = build(random_resolve(implement(task)), f'./outs/verify/host/{name}')

    GA, GB, GC = A.to_cuda(), B.to_cuda(), C.to_cuda()
    grid_module['matmul'](GA, GB, GC)

    HA, HB, HC = A.to_cpu(), B.to_cpu(), C.to_cpu()
    host_module['matmul'](HA, HB, HC)
    np.testing.assert_allclose(GC.to_numpy(), HC.to_numpy())


if __name__ == '__main__':
    pytest.main(__file__)
