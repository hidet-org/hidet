import torch
import pytest
import einops
from einops import EinopsError
from hidet.testing.torch_utils import check_module, FunctionalModule


@pytest.mark.parametrize('shape', [(30, 40, 4, 21)])
@pytest.mark.parametrize(
    "pattern, axes_lengths",
    [
        ('b h w c -> b h w c', {}),
        ('b h w c -> b c h w', {}),
        ('b h w c -> (b h) w c', {}),
        ('b h w c -> h (b w) c', {}),
        ('b h w c -> b (c h w)', {}),
        ('b (h1 h) (w1 w) c -> (b h1 w1) h w c', {'h1': 2, 'w1': 2}),
        ('b (h h1) (w w1) c -> b h w (c h1 w1)', {'h1': 2, 'w1': 2}),
        ('b (h h1) (w w1) ... -> b h w (... h1 w1)', {'h1': 2, 'w1': 2}),
        ('b (h h1) (w w1) ... -> b h w (h1 ... w1)', {'h1': 2, 'w1': 2}),
        ('b (h h1) (w w1) ... -> b h w (h1 w1 ...)', {'h1': 2, 'w1': 2}),
        ('b (h h1) (w w1) ... -> b h w ... (h1 w1)', {'h1': 2, 'w1': 2}),
        ('s b ... -> b s ...', {}),
        ('b s ... -> (b s) ...', {}),
        ('(b s) ... -> b s ...', {'b': 2}),
        ('b s h d -> s b (h d)', {}),
        ('b s h d -> b s h d', {}),
        ('b s ... -> (...) s b', {}),
    ],
)
def test_rearrange(shape, pattern, axes_lengths, device):
    print(f"\nTesting pattern: {pattern} with shape: {shape}")
    try:
        check_module(
            FunctionalModule(op=lambda x: einops.rearrange(x, pattern, **axes_lengths)),
            args=[torch.randn(shape)],
            device=device,
        )
    except EinopsError as e:  # If EinopsError is raised, skip the test
        print(e)
        pytest.skip(f"Skipping test for invalid pattern: {pattern} with shape: {shape}")


@pytest.mark.parametrize(
    'shape, pattern, axes_lengths',
    [
        ((1, 1, 1, 1, 1), '1 1 1 1 1 -> ', {}),
        ((30, 1, 4, 1, 1), 'b 1 h w 1 -> b h w', {}),
        ((30, 1, 4, 1, 1), 'b h w d 1 -> b 1 h w 1 d', {}),
        ((30, 1, 4, 21), 'b 1 h w -> b h w', {}),
        ((30, 1, 4, 21), 'b h w d -> b 1 h w 1 d', {}),
        # flatten the last two dims into one
        ((2, 3, 4), 'b h w      -> b (h w)', {}),
        # split the first dim (6) into b=2 and h=3
        ((6, 5, 4), '(b h) w d  -> b h w d', {'b': 2, 'h': 3}),
        # a straight permutation
        ((2, 3, 4, 5), 'b c h w    -> h w b c', {}),
        # move the last axis to the front via ellipsis
        ((2, 3, 4, 5), '... c      -> c ...', {}),
        # drop a literal‐1 axis
        ((2, 1, 3), 'b 1 h      -> b h', {}),
        # insert two length‐1 axes
        ((2, 3), 'b c        -> b 1 c 1', {}),
        # identity via ellipsis
        ((3, 4, 5), '...        -> ...', {}),
        # axis names with underscore are fine
        ((2, 3, 4), 'b c_d h    -> b h c_d', {}),
        ((2, 16, 80), '(b s) ... -> b s ...', {'b': 1}),
        ((2, 16, 80), '(b s) ... -> s b ...', {'b': 1}),
    ],
)
def test_custom(shape, pattern, axes_lengths, device):
    print(f"\nTesting pattern: {pattern} with shape: {shape}")
    try:
        check_module(
            FunctionalModule(op=lambda x: einops.rearrange(x, pattern, **axes_lengths)),
            args=[torch.randn(shape)],
            device=device,
        )
    except EinopsError as e:  # If EinopsError is raised, skip the test
        print(e)
        pytest.skip(f"Skipping test for invalid pattern: {pattern} with shape: {shape}")
