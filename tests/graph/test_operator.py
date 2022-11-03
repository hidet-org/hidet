import hidet
import pytest


def test_profile_config():
    a = hidet.randn([1, 10, 10], device='cuda')
    b = hidet.randn([1, 10, 10], device='cuda')
    hidet.option.search_space(1)
    hidet.option.bench_config(1, 1, 1)
    c = hidet.ops.batch_matmul(a, b)
    hidet.option.search_space(0)


if __name__ == '__main__':
    pytest.main(__file__)
