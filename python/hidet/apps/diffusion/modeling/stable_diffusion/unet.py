from typing import Callable, List, Optional, Tuple, Union
from hidet import nn
from hidet.apps.diffusion.modeling.pretrained import PretrainedModelForDiffusion
from hidet.apps.diffusion.modeling.stable_diffusion.timestep import TimestepEmbedding, Timesteps
from hidet.apps.diffusion.modeling.stable_diffusion.unet_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    MidBlock2DCrossAttn,
    UpBlock2D,
)
from hidet.apps.pretrained import PretrainedModel
from hidet.apps.registry import RegistryEntry
from hidet.graph.tensor import Tensor
from hidet.graph.ops import broadcast

import hidet

PretrainedModel.register(
    arch="UNet2DConditionModel",
    entry=RegistryEntry(model_category="diffusion", module_name="stable_diffusion", klass="UNet2DConditionModel"),
)


class UNet2DConditionModel(PretrainedModelForDiffusion):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        center_input_sample: bool,
        block_out_channels: List[int],
        conv_in_kernel: int,
        conv_out_kernel: int,
        down_block_types: List[str],
        mid_block_type: str,
        up_block_types: List[str],
        embed_max_length: int,
        embed_hidden_dim: int,
        vae_scale_factor: int,
        attention_head_dim: Union[int, Tuple[int, ...]],
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        cross_attention_dim: Union[int, Tuple[int, ...]] = 1024,
        only_cross_attention: Union[bool, Tuple[bool, ...]] = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        layers_per_block: Union[int, Tuple[int, ...]] = 2,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        time_embedding_type: str = "positional",
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        act_fn: Union[str, Callable] = "silu",
        norm_eps: float = 1e-05,
        norm_num_groups: int = 32,
        resnet_time_scale_shift: str = "default",
        downsample_padding: int = 1,
        use_linear_projection: bool = True,
        class_embeddings_concat: bool = False,
        dropout: float = 0.0,
        **kwargs,  # capture unnecessary kwargs
    ):
        super().__init__(
            {
                "in_channels": in_channels,
                "embed_max_length": embed_max_length,
                "embed_hidden_dim": embed_hidden_dim,
                "vae_scale_factor": vae_scale_factor,
                "center_input_sample": center_input_sample,
            }
        )
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=(conv_in_kernel - 1) // 2,
            bias=True,
        )

        assert time_embedding_type == "positional"
        timestep_input_dim = block_out_channels[0]
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)

        if isinstance(act_fn, str):
            act_fn = getattr(hidet.graph.ops, act_fn)

        self.time_embedding = TimestepEmbedding(
            in_channels=timestep_input_dim, time_embed_dim=time_embed_dim, act_fn=act_fn
        )

        self.down_blocks = []
        self.up_blocks = []

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention
            only_cross_attention = (only_cross_attention,) * len(down_block_types)

        mid_block_only_cross_attention = (
            mid_block_only_cross_attention if mid_block_only_cross_attention is not None else False
        )

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        num_attention_heads = num_attention_heads or attention_head_dim
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = (layers_per_block,) * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim
        if class_embeddings_concat:
            blocks_time_embed_dim *= 2

        output_channels = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channels = output_channels
            output_channels = block_out_channels[i]
            is_final = i == (len(block_out_channels) - 1)

            self.down_blocks.append(
                self.get_down_block(
                    down_block_type,
                    num_layers=layers_per_block[i],
                    input_channels=input_channels,
                    output_channels=output_channels,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final,
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    resnet_act_fn=act_fn,
                    resnet_eps=norm_eps,
                    resnet_groups=norm_num_groups,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    downsample_padding=downsample_padding,
                    use_linear_projection=use_linear_projection,
                    dropout=dropout,
                )
            )

        self.down_blocks = nn.ModuleList(self.down_blocks)

        assert mid_block_type == "UNetMidBlock2DCrossAttn"
        self.mid_block = MidBlock2DCrossAttn(
            num_layers=1,
            input_channels=block_out_channels[-1],
            output_channels=None,
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            resnet_act_fn=act_fn,
            resnet_eps=norm_eps,
            resnet_groups=norm_num_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_linear_projection=use_linear_projection,
            dropout=dropout,
        )

        self.num_upsamplers = 0

        r_block_out_channels = list(reversed(block_out_channels))
        r_num_attention_heads = list(reversed(num_attention_heads))
        r_cross_attention_dim = list(reversed(cross_attention_dim))
        r_layers_per_block = list(reversed(layers_per_block))
        r_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channels = r_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(r_block_out_channels) - 1

            prev_output_channels = output_channels
            output_channels = r_block_out_channels[i]
            input_channels = r_block_out_channels[min(i + 1, len(r_block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = self.get_up_block(
                up_block_type,
                num_layers=r_layers_per_block[i] + 1,
                input_channels=input_channels,
                output_channels=output_channels,
                temb_channels=blocks_time_embed_dim,
                transformer_layers_per_block=r_transformer_layers_per_block[i],
                prev_output_channel=prev_output_channels,
                add_upsample=add_upsample,
                resnet_act_fn=act_fn,
                resnet_eps=norm_eps,
                resnet_groups=norm_num_groups,
                num_attention_heads=r_num_attention_heads[i],
                cross_attention_dim=r_cross_attention_dim[i],
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_linear_projection=use_linear_projection,
                dropout=dropout,
            )

            self.up_blocks.append(up_block)
            prev_output_channels = output_channels

        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = act_fn

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding, bias=True
        )

    @property
    def embed_max_length(self):
        return self.config["embed_max_length"]

    @property
    def embed_hidden_dim(self):
        return self.config["embed_hidden_dim"]

    @property
    def vae_scale_factor(self):
        return self.config["vae_scale_factor"]

    def get_down_block(self, down_block_type: str, **kwargs):
        if down_block_type == "CrossAttnDownBlock2D":
            return CrossAttnDownBlock2D(**kwargs)  # type: ignore
        elif down_block_type == "DownBlock2D":
            return DownBlock2D(**kwargs)  # type: ignore
        else:
            raise ValueError(f"{down_block_type} not found.")

    def get_up_block(self, up_block_type: str, **kwargs):
        if up_block_type == "CrossAttnUpBlock2D":
            return CrossAttnUpBlock2D(**kwargs)
        elif up_block_type == "UpBlock2D":
            return UpBlock2D(**kwargs)
        else:
            raise ValueError(f"{up_block_type} not found.")

    def forward_down(self, sample: Tensor, timesteps: Tensor, encoder_hidden_states: Tensor) -> Tuple:
        if self.config["center_input_sample"]:
            sample = 2 * sample - 1.0

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        down_block_residual_samples = (sample,)
        for block in self.down_blocks:
            if block.has_cross_attention:
                sample, res_samples = block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)

            else:
                sample, res_samples = block(hidden_states=sample, temb=emb)

            down_block_residual_samples += res_samples

        return sample, emb, down_block_residual_samples

    def forward_mid(self, sample: Tensor, emb: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        if self.mid_block.has_cross_attention:
            return self.mid_block(sample, emb, encoder_hidden_states)
        else:
            return self.mid_block(sample, emb)

    def forward_up(
        self, sample: Tensor, emb: Tensor, encoder_hidden_states: Tensor, down_block_residual_samples: Tuple[Tensor]
    ) -> Tensor:
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        for i, block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_residual_samples[-len(block.resnets) :]
            down_block_residual_samples = down_block_residual_samples[: -len(block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_residual_samples[-1].shape[2:]

            if block.has_cross_attention:
                sample = block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=emb,
                    upsample_size=upsample_size,
                    encoder_hidden_states=encoder_hidden_states,
                    is_final_block=is_final_block,
                )
            else:
                sample = block(
                    hidden_states=sample, res_hidden_states_tuple=res_samples, temb=emb, upsample_size=upsample_size
                )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    def forward(self, sample: Tensor, timesteps: Tensor, encoder_hidden_states: Tensor, **kwargs) -> Tensor:
        timesteps = broadcast(timesteps, shape=(sample.shape[0],))

        sample, emb, down_block_residual_samples = self.forward_down(sample, timesteps, encoder_hidden_states)

        sample = self.forward_mid(sample, emb, encoder_hidden_states)

        sample = self.forward_up(sample, emb, encoder_hidden_states, down_block_residual_samples)

        return sample
