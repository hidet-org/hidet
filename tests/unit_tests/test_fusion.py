import hidet


def test_fusion_v1():
    @hidet.jit(opt=True)
    def func(a: hidet.Tensor, b: hidet.Tensor):
        c = hidet.ops.equal(a, b)
        d = hidet.ops.logical_not(c)
        e = d.cast('int32')
        f = hidet.ops.cumsum(e, dim=1)
        g = f * e
        h = g.cast('int64')
        i = h + 1
        return i

    a = hidet.zeros([1, 9], dtype='int64')
    b = hidet.zeros([], dtype='int64')
    func(a, b)
