from hidet.apps.diffusion.app import DiffusionApp
from hidet.apps.diffusion.builder import create_stable_diffusion
import pytest


def test_create_stable_diffusion():
    diffusion_app: DiffusionApp = create_stable_diffusion(
        "stabilityai/stable-diffusion-2-1", kernel_search_space=0, height=512, width=512
    )
    res = diffusion_app.generate_image(
        "Software engineer writing code at desk with laptop, soft glow, detailed image.", "blurry, multiple fingers"
    )

    assert res[0].height == 512
    assert res[0].width == 512
    assert res[0].mode == "RGB"


if __name__ == "__main__":
    pytest.main([__file__])
