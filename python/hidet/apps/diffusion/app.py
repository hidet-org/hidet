import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from hidet.graph.tensor import from_torch, full
from hidet.runtime.compiled_app import CompiledApp


class DiffusionApp:
    def __init__(self, compiled_app: CompiledApp, hf_pipeline: StableDiffusionPipeline, height: int, width: int):
        super().__init__()
        assert height % 8 == 0 and width % 8 == 0, "Height and width must be multiples of 8"
        self.height = height
        self.width = width
        self.compiled_app: CompiledApp = compiled_app
        self.hf_pipeline: StableDiffusionPipeline = hf_pipeline

        self.hf_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.hf_pipeline.scheduler.config)
        self.hf_pipeline = self.hf_pipeline.to("cuda")

        def _unet_forward(sample: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states: torch.Tensor, **kwargs):
            h_sample = from_torch(sample)
            h_timesteps = full([sample.shape[0]], timesteps.item(), dtype="int64", device="cuda")
            h_encoder_hidden_states = from_torch(encoder_hidden_states)

            down_outs = self.compiled_app.graphs["unet_down"](h_sample, h_timesteps, h_encoder_hidden_states)
            h_sample = down_outs[0]
            h_emb = down_outs[1]
            h_down_block_residual_samples = down_outs[2:]

            h_sample = self.compiled_app.graphs["unet_mid"](h_sample, h_emb, h_encoder_hidden_states)

            h_sample = self.compiled_app.graphs["unet_up"](
                h_sample, h_emb, h_encoder_hidden_states, *h_down_block_residual_samples
            )

            return (h_sample.torch(),)

        self.hf_pipeline.unet.forward = _unet_forward

    def generate_image(self, prompt: str, negative_prompt: str):

        return self.hf_pipeline(
            prompt=prompt, negative_prompt=negative_prompt, height=self.height, width=self.width
        ).images
