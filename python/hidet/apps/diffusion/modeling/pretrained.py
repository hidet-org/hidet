from typing import Optional
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

from hidet.apps.hf import load_diffusion_pipeline
from hidet.apps.pretrained import PretrainedModel


class PretrainedModelForDiffusion(PretrainedModel):
    @classmethod
    def create_pretrained_model(
        cls,
        name: str,
        revision: Optional[str] = None,
        hf_pipeline: Optional[StableDiffusionPipeline] = None,
        dtype: Optional[str] = None,
        device: str = "cuda",
    ):
        # load the pretrained huggingface model
        # note: diffusers pipeline is more similar to a model than transformers pipeline
        if hf_pipeline is None:
            hf_pipeline: StableDiffusionPipeline = load_diffusion_pipeline(name=name, revision=revision, device=device)

        pipeline_config = hf_pipeline.config

        torch_unet = hf_pipeline.unet
        pretrained_unet_class = cls.load_module(pipeline_config["unet"][1])
        hf_config = dict(torch_unet.config)

        if not all(
            x is None
            for x in (
                hf_config["encoder_hid_dim_type"],
                hf_config["class_embed_type"],
                hf_config["addition_embed_type"],
            )
        ):
            raise NotImplementedError("Additional projection and embedding features not included yet.")

        hidet_unet = pretrained_unet_class(
            **hf_config,
            vae_scale_factor=hf_pipeline.vae_scale_factor,
            embed_max_length=hf_pipeline.text_encoder.config.max_position_embeddings,
            embed_hidden_dim=hf_pipeline.text_encoder.config.hidden_size
        )

        hidet_unet.to(dtype=dtype, device=device)

        cls.copy_weights(torch_unet, hidet_unet)

        return hidet_unet

    @property
    def embed_max_length(self):
        raise NotImplementedError()

    @property
    def embed_hidden_dim(self):
        raise NotImplementedError()

    @property
    def vae_scale_factor(self):
        raise NotImplementedError()

    def forward_down(self, *args, **kwargs):
        raise NotImplementedError()

    def forward_mid(self, *args, **kwargs):
        raise NotImplementedError()

    def forward_up(self, *args, **kwargs):
        raise NotImplementedError()
