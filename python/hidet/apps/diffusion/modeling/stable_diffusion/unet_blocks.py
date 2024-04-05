from typing import Tuple
from hidet import nn
from hidet.apps.diffusion.modeling.stable_diffusion.downsample import Downsample2D
from hidet.apps.diffusion.modeling.stable_diffusion.resnet_blocks import ResnetBlock2D
from hidet.apps.diffusion.modeling.stable_diffusion.transformer_blocks import Transformer2DModel
from hidet.apps.diffusion.modeling.stable_diffusion.upsample import Upsample2D
from hidet.graph.tensor import Tensor
from hidet.graph.ops import concat


class CrossAttnDownBlock2D(nn.Module[Tensor]):
    def __init__(self, **kwargs):
        super().__init__()
        self.has_cross_attention = True
        self.resnets = []
        self.attentions = []

        transformer_layers_per_block = kwargs["transformer_layers_per_block"]
        num_layers = kwargs["num_layers"]

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            input_channels = kwargs["input_channels"] if i == 0 else kwargs["output_channels"]
            self.resnets.append(ResnetBlock2D(**{**kwargs, "input_channels": input_channels}))
            self.attentions.append(
                Transformer2DModel(
                    **{
                        **kwargs,
                        "attention_head_dim": kwargs["output_channels"] // kwargs["num_attention_heads"],
                        "input_channels": kwargs["output_channels"],
                        "num_layers": transformer_layers_per_block[i],
                    }
                )
            )

        self.resnets = nn.ModuleList(self.resnets)
        self.attentions = nn.ModuleList(self.attentions)

        if kwargs["add_downsample"]:
            self.downsamplers = nn.ModuleList([Downsample2D(kwargs["output_channels"], **kwargs)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: Tensor, temb: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        output_states = ()
        blocks = list(zip(self.resnets, self.attentions))

        for resnet, attn in blocks:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(nn.Module[Tensor]):
    def __init__(self, **kwargs):
        super().__init__()
        self.has_cross_attention = False
        self.resnets = []

        for i in range(kwargs["num_layers"]):
            input_channels = kwargs["input_channels"] if i == 0 else kwargs["output_channels"]
            self.resnets.append(ResnetBlock2D(**{**kwargs, "input_channels": input_channels}))

        self.resnets = nn.ModuleList(self.resnets)
        if kwargs["add_downsample"]:
            self.downsamplers = nn.ModuleList([Downsample2D(kwargs["output_channels"], **kwargs)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class MidBlock2DCrossAttn(nn.Module[Tensor]):
    def __init__(self, **kwargs):
        super().__init__()

        self.has_cross_attention = True

        transformer_layers_per_block = kwargs["transformer_layers_per_block"]
        if isinstance(kwargs["transformer_layers_per_block"], int):
            transformer_layers_per_block = [transformer_layers_per_block] * kwargs["num_layers"]

        self.resnets = [ResnetBlock2D(**{**kwargs, "input_channels": kwargs["input_channels"]})]
        self.attentions = []

        for i in range(kwargs["num_layers"]):
            self.attentions.append(
                Transformer2DModel(
                    **{
                        **kwargs,
                        "attention_head_dim": kwargs["input_channels"] // kwargs["num_attention_heads"],
                        "input_channels": kwargs["input_channels"],
                        "num_layers": transformer_layers_per_block[i],
                    }
                )
            )

            self.resnets.append(ResnetBlock2D(**{**kwargs, "input_channels": kwargs["input_channels"]}))

        self.resnets = nn.ModuleList(self.resnets)
        self.attentions = nn.ModuleList(self.attentions)

    def forward(self, hidden_states: Tensor, temb: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CrossAttnUpBlock2D(nn.Module[Tensor]):
    def __init__(self, **kwargs):
        super().__init__()
        self.has_cross_attention = True
        num_layers = kwargs["num_layers"]

        transformer_layers_per_block = kwargs["transformer_layers_per_block"]
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        self.resnets = []
        self.attentions = []
        for i in range(num_layers):
            res_skip_channels = kwargs["input_channels"] if (i == num_layers - 1) else kwargs["output_channels"]
            resnet_in_channels = kwargs["prev_output_channel"] if i == 0 else kwargs["output_channels"]
            input_channels = resnet_in_channels + res_skip_channels

            self.resnets.append(ResnetBlock2D(**{**kwargs, "input_channels": input_channels}))

            self.attentions.append(
                Transformer2DModel(
                    **{
                        **kwargs,
                        "attention_head_dim": kwargs["output_channels"] // kwargs["num_attention_heads"],
                        "input_channels": kwargs["output_channels"],
                        "num_layers": transformer_layers_per_block[i],
                    }
                )
            )

        self.resnets = nn.ModuleList(self.resnets)
        self.attentions = nn.ModuleList(self.attentions)

        if kwargs["add_upsample"]:
            self.upsamplers = nn.ModuleList([Upsample2D(kwargs["output_channels"], **kwargs)])
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor],
        temb: Tensor,
        upsample_size: int,
        encoder_hidden_states: Tensor,
        is_final_block=False,
    ) -> Tensor:
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = concat([hidden_states, res_hidden_states], axis=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temperature_scaling=2 if is_final_block and i == 1 else 1,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock2D(nn.Module[Tensor]):
    def __init__(self, **kwargs):
        super().__init__()
        self.has_cross_attention = False
        self.resnets = []

        for i in range(kwargs["num_layers"]):
            res_skip_channels = (
                kwargs["input_channels"] if (i == kwargs["num_layers"] - 1) else kwargs["output_channels"]
            )
            resnet_input_channels = kwargs["prev_output_channel"] if i == 0 else kwargs["output_channels"]
            input_channels = res_skip_channels + resnet_input_channels

            self.resnets.append(ResnetBlock2D(**{**kwargs, "input_channels": input_channels}))

        self.resnets = nn.ModuleList(self.resnets)
        if kwargs["add_upsample"]:
            self.upsamplers = nn.ModuleList([Upsample2D(kwargs["output_channels"], **kwargs)])
        else:
            self.upsamplers = None

    def forward(
        self, hidden_states: Tensor, res_hidden_states_tuple: Tuple[Tensor], temb: Tensor, upsample_size: int
    ) -> Tensor:
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = concat([hidden_states, res_hidden_states], axis=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
