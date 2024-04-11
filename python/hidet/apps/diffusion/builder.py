from typing import Optional, Tuple
from diffusers import StableDiffusionPipeline

import hidet
from hidet.apps.diffusion.app import DiffusionApp
from hidet.apps.diffusion.modeling.pretrained import PretrainedModelForDiffusion
from hidet.apps.hf import load_diffusion_pipeline
from hidet.graph.flow_graph import trace_from
from hidet.graph import FlowGraph
from hidet.graph.tensor import symbol, Tensor
from hidet.runtime.compiled_app import create_compiled_app


def _build_unet_down_graph(
    model: PretrainedModelForDiffusion,
    dtype: str = "float32",
    device: str = "cuda",
    batch_size: int = 2,
    num_channels_latents: int = 4,
    height: int = 96,
    width: int = 96,
    embed_length: int = 77,
    embed_hidden_dim: int = 1024,
    kernel_search_space: int = 2,
):
    latent_model_input: Tensor = symbol([batch_size, num_channels_latents, height, width], dtype=dtype, device=device)
    timesteps: Tensor = symbol([batch_size], dtype="int64", device=device)
    prompt_embeds: Tensor = symbol([batch_size, embed_length, embed_hidden_dim], dtype=dtype, device=device)

    inputs = (latent_model_input, timesteps, prompt_embeds)
    outputs = sample, emb, down_block_residual_samples = model.forward_down(*inputs)
    graph: FlowGraph = trace_from([sample, emb, *down_block_residual_samples], list(inputs))

    graph = hidet.graph.optimize(graph)

    compiled_graph = graph.build(space=kernel_search_space)

    return compiled_graph, inputs, outputs


def _build_unet_mid_graph(
    model: PretrainedModelForDiffusion,
    sample: Tensor,
    emb: Tensor,
    encoder_hidden_states: Tensor,
    kernel_search_space: int = 2,
):
    sample, emb, encoder_hidden_states = tuple(
        symbol(list(x.shape), dtype=x.dtype, device=x.device) for x in (sample, emb, encoder_hidden_states)
    )

    output = model.forward_mid(sample, emb, encoder_hidden_states)

    graph: FlowGraph = trace_from(output, [sample, emb, encoder_hidden_states])

    graph = hidet.graph.optimize(graph)

    compiled_graph = graph.build(space=kernel_search_space)

    return compiled_graph, output


def _build_unet_up_graph(
    model: PretrainedModelForDiffusion,
    sample: Tensor,
    emb: Tensor,
    encoder_hidden_states: Tensor,
    down_block_residuals: Tuple[Tensor, ...],
    kernel_search_space: int = 2,
):
    sample, emb, encoder_hidden_states = tuple(
        symbol(list(x.shape), dtype=x.dtype, device=x.device) for x in (sample, emb, encoder_hidden_states)
    )
    down_block_residuals = tuple(symbol(list(x.shape), dtype=x.dtype, device=x.device) for x in down_block_residuals)
    output = model.forward_up(sample, emb, encoder_hidden_states, down_block_residuals)

    graph: FlowGraph = trace_from(output, [sample, emb, encoder_hidden_states, *down_block_residuals])

    graph = hidet.graph.optimize(graph)

    compiled_graph = graph.build(space=kernel_search_space)

    return compiled_graph, output


def create_stable_diffusion(
    name: str,
    revision: Optional[str] = None,
    dtype: str = "float32",
    device: str = "cuda",
    batch_size: int = 1,
    height: int = 768,
    width: int = 768,
    kernel_search_space: int = 2,
):
    hf_pipeline: StableDiffusionPipeline = load_diffusion_pipeline(name=name, revision=revision, device=device)
    # create the hidet model and load the pretrained weights from huggingface
    model: PretrainedModelForDiffusion = PretrainedModelForDiffusion.create_pretrained_model(
        name, revision=revision, device=device, dtype=dtype
    )

    unet_down_graph, inputs, outputs = _build_unet_down_graph(
        model,
        dtype=dtype,
        device=device,
        batch_size=batch_size * 2,  # double size for prompt/negative prompt
        num_channels_latents=model.config["in_channels"],
        height=height // model.vae_scale_factor,
        width=width // model.vae_scale_factor,
        embed_length=model.embed_max_length,
        embed_hidden_dim=model.embed_hidden_dim,
        kernel_search_space=kernel_search_space,
    )

    _, _, prompt_embeds = inputs
    sample, emb, down_block_residual_samples = outputs

    unet_mid_graph, sample = _build_unet_mid_graph(
        model, sample=sample, emb=emb, encoder_hidden_states=prompt_embeds, kernel_search_space=kernel_search_space
    )

    unet_up_graph, sample = _build_unet_up_graph(
        model,
        sample=sample,
        emb=emb,
        encoder_hidden_states=prompt_embeds,
        down_block_residuals=down_block_residual_samples,
        kernel_search_space=kernel_search_space,
    )

    return DiffusionApp(
        compiled_app=create_compiled_app(
            graphs={"unet_down": unet_down_graph, "unet_mid": unet_mid_graph, "unet_up": unet_up_graph},
            modules={},
            tensors={},
            attributes={},
            name=name,
        ),
        hf_pipeline=hf_pipeline,
        height=height,
        width=width,
    )
