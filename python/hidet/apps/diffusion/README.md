## Hidet Stable Diffusion Compiled App

### Quickstart

```
from hidet.apps.diffusion.app import DiffusionApp
from hidet.apps.diffusion.builder import create_stable_diffusion

diffusion_app: DiffusionApp = create_stable_diffusion(
    "stabilityai/stable-diffusion-2-1", height=512, width=512, kernel_search_space=0
)

res = diffusion_app.generate_image(
    "Software engineer works on code in front of monitor at desk at night with lamp lighting, soft glow, 4k detailed",
    "blurry, too many fingers",
)

res[0].save("diffusion_app_example.png")
```

A stable diffusion app can be constructed using any Huggingface model identifier that uses the StableDiffusion v2-1 architecture. Currently supported model inputs are a prompt/negative-prompt. The app produces a PIL image, which can then be saved to disk. Use kernel_search_space=2 for optimization.

If the weights used are not public, be sure to modify `hidet.toml` so that option `auth_tokens.for_huggingface` is set to your Huggingface account credential.

### Model Details

A `PretrainedModelForDiffusion` is a `PretrainedModel` that allows us to `create_pretrained_model` from a Huggingface identifier. Currently only the UNet portion of stable diffusion is entirely implemented in Hidet (the remaining VAE, decoder, tokenizer, etc. are relatively fast). `PretrainedModelForDiffusion` defines a forward function for each of the down/mid/up segments of a UNet, where the actual diffusion steps take place. 

`UNet2DConditionModel`, a child class of `PretrainedModelForDiffusion`, accepts the Huggingface stable diffusion config as a `dict` and defines the model architecture and feed-forward steps. This Hidet implementation is directly injected into the Huggingface pipeline by replacing the forward call to its UNet (see `./app.py`). Therefore a `DiffusionApp` is currently a wrapper with an interface for interacting with the Huggingface pipeline. Once the rest of the components are included, the `DiffusionApp` should no longer have the Huggingface library as a dependency.

Note: Currently Hidet uses flash attention with fp16, which produces overflow for large magnitudes given the current weights. The Pytorch scaled dot product attention uses fp32. As such, the Hidet implementation scales the input to the softmax by 0.5 for the layer this happens (see `temperature_scaling` in `transformer_blocks.py`), producing a slight error relative to the original. This does not seem to have observable impact on the final outputs.
