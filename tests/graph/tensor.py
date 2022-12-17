import pytest
import numpy as np
import torch
import hidet


def test_from_dlpack():
    a = torch.randn([2, 3])
    b = hidet.from_dlpack(a)
    np.testing.assert_allclose(a.numpy(), b.numpy())

    a[1:] = 1.0
    np.testing.assert_allclose(a.numpy(), b.numpy())

    c = torch.randn([2, 3]).cuda()
    d = hidet.from_dlpack(c)
    np.testing.assert_allclose(c.cpu().numpy(), d.cpu().numpy())

    c[1:] = 1.0
    np.testing.assert_allclose(c.cpu().numpy(), d.cpu().numpy())


def test_to_dlpack():
    a = hidet.randn([2, 3]).cpu()
    b = torch.from_dlpack(a)
    np.testing.assert_allclose(a.numpy(), b.numpy())

    b[1:] = 1.0
    np.testing.assert_allclose(a.numpy(), b.numpy())

    d = hidet.randn([2, 3]).cuda()
    e = torch.from_dlpack(d)
    np.testing.assert_allclose(d.cpu().numpy(), e.cpu().numpy())

    e[1:] = 1.0
    np.testing.assert_allclose(d.cpu().numpy(), e.cpu().numpy())
