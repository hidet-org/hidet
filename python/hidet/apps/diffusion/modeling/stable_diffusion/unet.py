from typing import Tuple
from hidet import nn
from hidet.apps.diffusion.modeling.pretrained import PretrainedModelForText2Image
from hidet.apps.diffusion.modeling.stable_diffusion.timestep import TimestepEmbedding, Timesteps
from hidet.apps.diffusion.modeling.stable_diffusion.unet_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    MidBlock2DCrossAttn,
    UpBlock2D,
)
from hidet.apps.modeling_outputs import UNet2DConditionOutput
from hidet.apps.pretrained import PretrainedModel
from hidet.apps.registry import RegistryEntry
from hidet.graph.tensor import Tensor
from hidet.graph.ops import broadcast

import hidet

PretrainedModel.register(
    arch="UNet2DConditionModel",
    entry=RegistryEntry(model_category="diffusion", module_name="stable_diffusion", klass="UNet2DConditionModel"),
)


class UNet2DConditionModel(PretrainedModelForText2Image):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.conv_in = nn.Conv2d(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["block_out_channels"][0],
            kernel_size=kwargs["conv_in_kernel"],
            padding=(kwargs["conv_in_kernel"] - 1) // 2,
            bias=True,
        )

        assert kwargs["time_embedding_type"] == "positional"
        timestep_input_dim = kwargs["block_out_channels"][0]
        time_embed_dim = kwargs["block_out_channels"][0] * 4

        self.time_proj = Timesteps(kwargs["block_out_channels"][0], kwargs["flip_sin_to_cos"], kwargs["freq_shift"])

        kwargs["act_fn"] = getattr(hidet.graph.ops, kwargs["act_fn"])
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=kwargs["act_fn"])

        if not all(
            x is None
            for x in (kwargs["encoder_hid_dim_type"], kwargs["class_embed_type"], kwargs["addition_embed_type"])
        ):
            raise NotImplementedError("Additional projection and embedding features not included yet.")

        self.down_blocks = []
        self.up_blocks = []

        down_block_types = kwargs["down_block_types"]
        only_cross_attention = kwargs["only_cross_attention"]
        mid_block_only_cross_attention = kwargs["mid_block_only_cross_attention"]

        if isinstance(kwargs["only_cross_attention"], bool):
            if kwargs["mid_block_only_cross_attention"] is None:
                mid_block_only_cross_attention = only_cross_attention
            only_cross_attention = [only_cross_attention] * len(down_block_types)  # 4

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        attention_head_dim = kwargs["attention_head_dim"]
        if isinstance(kwargs["attention_head_dim"], int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        num_attention_heads = kwargs["num_attention_heads"] or attention_head_dim
        if isinstance(kwargs["num_attention_heads"], int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        cross_attention_dim = kwargs["cross_attention_dim"]
        if isinstance(kwargs["cross_attention_dim"], int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        layers_per_block = kwargs["layers_per_block"]
        if isinstance(kwargs["layers_per_block"], int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        transformer_layers_per_block = kwargs["transformer_layers_per_block"]
        if isinstance(kwargs["transformer_layers_per_block"], int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim
        if kwargs["class_embeddings_concat"]:
            blocks_time_embed_dim *= 2

        output_channels = kwargs["block_out_channels"][0]
        for i, down_block_type in enumerate(kwargs["down_block_types"]):
            input_channels = output_channels
            output_channels = kwargs["block_out_channels"][i]
            is_final = i == (len(kwargs["block_out_channels"]) - 1)

            self.down_blocks.append(
                self.get_down_block(
                    down_block_type,
                    num_layers=layers_per_block[i],
                    input_channels=input_channels,
                    output_channels=output_channels,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    resnet_eps=kwargs["norm_eps"],
                    resnet_act_fn=kwargs["act_fn"],
                    resnet_groups=kwargs["norm_num_groups"],
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    only_cross_attention=only_cross_attention[i],
                    attention_head_dim=attention_head_dim[i] or output_channels,
                )
            )

        self.down_blocks = nn.ModuleList(self.down_blocks)

        self.mid_block = MidBlock2DCrossAttn(
            **{
                **self.config,
                "input_channels": kwargs["block_out_channels"][-1],
                "output_channels": None,
                "temb_channels": blocks_time_embed_dim,
                "num_layers": 1,
                "transformer_layers_per_block": transformer_layers_per_block[-1],
                "resnet_eps": kwargs["norm_eps"],
                "resnet_act_fn": kwargs["act_fn"],
                "resnet_groups": kwargs["norm_num_groups"],
                "cross_attention_dim": cross_attention_dim[-1],
                "num_attention_heads": num_attention_heads[-1],
            }
        )

        self.num_upsamplers = 0

        r_block_out_channels = list(reversed(kwargs["block_out_channels"]))
        r_num_attention_heads = list(reversed(num_attention_heads))
        r_layers_per_block = list(reversed(layers_per_block))
        r_cross_attention_dim = list(reversed(cross_attention_dim))
        r_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        r_only_cross_attention = list(reversed(only_cross_attention))

        output_channels = r_block_out_channels[0]
        for i, up_block_type in enumerate(kwargs["up_block_types"]):
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
                transformer_layers_per_block=r_transformer_layers_per_block[i],
                input_channels=input_channels,
                output_channels=output_channels,
                prev_output_channel=prev_output_channels,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=kwargs["norm_eps"],
                resnet_act_fn=kwargs["act_fn"],
                resnet_groups=kwargs["norm_num_groups"],
                resolution_idx=i,
                cross_attention_dim=r_cross_attention_dim[i],
                num_attention_heads=r_num_attention_heads[i],
                only_cross_attention=r_only_cross_attention[i],
                attention_head_dim=(attention_head_dim[i] if attention_head_dim[i] is not None else output_channels),
            )

            self.up_blocks.append(up_block)
            prev_output_channels = output_channels

        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.conv_norm_out = nn.GroupNorm(
            num_channels=kwargs["block_out_channels"][0], num_groups=kwargs["norm_num_groups"], eps=kwargs["norm_eps"]
        )
        self.conv_act = kwargs["act_fn"]

        conv_out_padding = (kwargs["conv_out_kernel"] - 1) // 2
        self.conv_out = nn.Conv2d(
            kwargs["block_out_channels"][0],
            kwargs["out_channels"],
            kernel_size=kwargs["conv_out_kernel"],
            padding=conv_out_padding,
            bias=True,
        )

    def get_down_block(self, down_block_type: str, **kwargs):
        if down_block_type == "CrossAttnDownBlock2D":
            return CrossAttnDownBlock2D(**{**self.config, **kwargs})  # type: ignore
        elif down_block_type == "DownBlock2D":
            return DownBlock2D(**{**self.config, **kwargs})  # type: ignore
        else:
            raise ValueError(f"{down_block_type} not found.")

    def get_up_block(self, up_block_type: str, **kwargs):
        if up_block_type == "CrossAttnUpBlock2D":
            return CrossAttnUpBlock2D(**{**self.config, **kwargs})
        elif up_block_type == "UpBlock2D":
            return UpBlock2D(**{**self.config, **kwargs})
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

    def forward(
        self, sample: Tensor, timesteps: Tensor, encoder_hidden_states: Tensor, **kwargs
    ) -> UNet2DConditionOutput:
        timesteps = broadcast(timesteps, shape=(sample.shape[0],))

        sample, emb, down_block_residual_samples = self.forward_down(sample, timesteps, encoder_hidden_states)

        sample = self.forward_mid(sample, emb, encoder_hidden_states)

        sample = self.forward_up(sample, emb, encoder_hidden_states, down_block_residual_samples)

        return UNet2DConditionOutput(last_hidden_state=sample, hidden_states=[sample])
