import numpy as np
import pytest

from hidet.backend import build
from hidet.baselines.matmul import matmul_opt
from hidet.implement import implement, impl_context
from hidet.implement.cuda.matmul import CudaGridStaticMatmulImplementer
from hidet.implement.resolve import random_resolve
from hidet.ir.task import Grid, Host
from hidet.tasks.nn import matmul
from hidet.tos.tensor import randn, zeros


@pytest.mark.parametrize('N,M,K,name,packed_func', [
    [256, 256, 256, 'opt', matmul_opt()]
])
def test_baseline(N, M, K, name, packed_func):
    task = matmul(N, M, K)
    A = randn([N, K], 'float32', device='cpu')
    B = randn([K, M], 'float32', device='cpu')
    C = zeros([N, M], 'float32', device='cpu')

    task.worker = Host()
    host_module = build(random_resolve(implement(task)), f'./outs/verify/host/{name}')

    GA, GB, GC = A.cuda(), B.cuda(), C.cuda()
    packed_func(N, M, K, GA, GB, GC)

    HA, HB, HC = A.cpu(), B.cpu(), C.cpu()
    host_module['matmul'](HA, HB, HC)
    np.testing.assert_allclose(GC.cpu().numpy(), HC.cpu().numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('N,M,K,name,implementers', [
    (234, 345, 567, 'HidetMatmul', (CudaGridStaticMatmulImplementer,)),
])
def test_hidet_variant(N, M, K, name, implementers):
    task = matmul(N, M, K)
    A = randn([N, K], 'float32', device='cpu')
    B = randn([K, M], 'float32', device='cpu')
    C = zeros([N, M], 'float32', device='cpu')

    task.worker = Grid()
    with impl_context(allowed=implementers):
        ir_module = implement(task)
        grid_module = build(random_resolve(ir_module, seed=1), f'./outs/verify/{name}')

    task.worker = Host()
    host_module = build(random_resolve(implement(task)), f'./outs/verify/host/{name}')

    GA, GB, GC = A.cuda(), B.cuda(), C.cuda()
    grid_module['matmul'](GA, GB, GC)

    HA, HB, HC = A.cpu(), B.cpu(), C.cpu()
    host_module['matmul'](HA, HB, HC)
    np.testing.assert_allclose(GC.cpu().numpy(), HC.cpu().numpy(), rtol=1e-5, atol=1e-4)


if __name__ == '__main__':
    pytest.main(__file__)
