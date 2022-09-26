import hidet
import pytest


def test_profile_config():
    a = hidet.randn([10, 10], device='cuda')
    b = hidet.randn([10, 10], device='cuda')
    hidet.space_level(2)
    hidet.profile_config(1, 1, 1)
    c = hidet.ops.matmul(a, b)


if __name__ == '__main__':
    pytest.main(__file__)
