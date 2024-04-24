from typing import Callable, Optional, Tuple, Union
from hidet import nn
from hidet.apps.diffusion.modeling.stable_diffusion.downsample import Downsample2D
from hidet.apps.diffusion.modeling.stable_diffusion.resnet_blocks import ResnetBlock2D
from hidet.apps.diffusion.modeling.stable_diffusion.transformer_blocks import Transformer2DModel
from hidet.apps.diffusion.modeling.stable_diffusion.upsample import Upsample2D
from hidet.graph.tensor import Tensor
from hidet.graph.ops import concat


class CrossAttnDownBlock2D(nn.Module[Tensor]):
    def __init__(
        self,
        *,
        num_layers: int,
        input_channels: int,
        output_channels: int,
        temb_channels: int,
        add_downsample: bool,
        num_attention_heads: int,
        cross_attention_dim: int,
        transformer_layers_per_block: Union[int, Tuple[int, ...]],
        resnet_act_fn: Callable,
        resnet_eps: float,
        resnet_groups: int,
        resnet_time_scale_shift: str,
        downsample_padding: int,
        use_linear_projection: bool,
        dropout: float,
        **kwargs,
    ):
        super().__init__()
        self.has_cross_attention = True
        self.resnets = []
        self.attentions = []

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * num_layers

        for i in range(num_layers):
            resnet_input_channels = input_channels if i == 0 else output_channels
            self.resnets.append(
                ResnetBlock2D(
                    input_channels=resnet_input_channels,
                    output_channels=output_channels,
                    resnet_groups=resnet_groups,
                    temb_channels=temb_channels,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=resnet_act_fn,
                    resnet_eps=resnet_eps,
                    dropout=dropout,
                )
            )
            self.attentions.append(
                Transformer2DModel(
                    num_layers=transformer_layers_per_block[i],
                    input_channels=output_channels,
                    output_channels=output_channels,
                    attention_head_dim=output_channels // num_attention_heads,
                    num_attention_heads=num_attention_heads,
                    cross_attention_dim=cross_attention_dim,
                    resnet_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.resnets = nn.ModuleList(self.resnets)
        self.attentions = nn.ModuleList(self.attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(output_channels, output_channels, downsample_padding)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: Tensor, temb: Tensor, encoder_hidden_states: Tensor) -> Tuple:
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
    def __init__(
        self,
        *,
        num_layers: int,
        input_channels: int,
        output_channels: int,
        temb_channels: int,
        add_downsample: bool,
        resnet_act_fn: Callable,
        resnet_eps: float,
        resnet_groups: int,
        resnet_time_scale_shift: str,
        downsample_padding: int,
        dropout: float,
        **kwargs,
    ):
        super().__init__()
        self.has_cross_attention = False
        self.resnets = []

        for i in range(num_layers):
            resnet_input_channels = input_channels if i == 0 else output_channels
            self.resnets.append(
                ResnetBlock2D(
                    input_channels=resnet_input_channels,
                    output_channels=output_channels,
                    resnet_groups=resnet_groups,
                    temb_channels=temb_channels,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=resnet_act_fn,
                    resnet_eps=resnet_eps,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(self.resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(output_channels, output_channels, downsample_padding)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tuple:
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
    def __init__(
        self,
        *,
        num_layers: int,
        input_channels: int,
        output_channels: Optional[int],
        temb_channels: int,
        transformer_layers_per_block: Union[int, Tuple[int, ...]],
        num_attention_heads: int,
        cross_attention_dim: int,
        resnet_act_fn: Callable,
        resnet_eps: float,
        resnet_groups: int,
        resnet_time_scale_shift: str,
        use_linear_projection: bool,
        dropout: float,
        **kwargs,
    ):
        super().__init__()

        self.has_cross_attention = True

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * num_layers

        self.resnets = [
            ResnetBlock2D(
                input_channels=input_channels,
                output_channels=output_channels,
                resnet_groups=resnet_groups,
                temb_channels=temb_channels,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_act_fn=resnet_act_fn,
                resnet_eps=resnet_eps,
                dropout=dropout,
            )
        ]
        self.attentions = []

        for i in range(num_layers):
            self.attentions.append(
                Transformer2DModel(
                    num_layers=transformer_layers_per_block[i],
                    input_channels=input_channels,
                    output_channels=output_channels,
                    attention_head_dim=input_channels // num_attention_heads,
                    num_attention_heads=num_attention_heads,
                    cross_attention_dim=cross_attention_dim,
                    resnet_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                )
            )

            self.resnets.append(
                ResnetBlock2D(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    resnet_groups=resnet_groups,
                    temb_channels=temb_channels,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=resnet_act_fn,
                    resnet_eps=resnet_eps,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(self.resnets)
        self.attentions = nn.ModuleList(self.attentions)

    def forward(self, hidden_states: Tensor, temb: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CrossAttnUpBlock2D(nn.Module[Tensor]):
    def __init__(
        self,
        *,
        num_layers: int,
        input_channels: int,
        output_channels: int,
        temb_channels: int,
        transformer_layers_per_block: Union[int, Tuple[int, ...]],
        prev_output_channel: int,
        add_upsample: bool,
        resnet_act_fn: Callable,
        resnet_groups: int,
        resnet_eps: float,
        resnet_time_scale_shift: str,
        num_attention_heads: int,
        cross_attention_dim: int,
        use_linear_projection: bool,
        dropout: float,
        **kwargs,
    ):
        super().__init__()
        self.has_cross_attention = True

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * num_layers

        self.resnets = []
        self.attentions = []
        for i in range(num_layers):
            res_skip_channels = input_channels if (i == num_layers - 1) else output_channels
            resnet_in_channels = prev_output_channel if i == 0 else output_channels
            resnet_input_channels = resnet_in_channels + res_skip_channels

            self.resnets.append(
                ResnetBlock2D(
                    input_channels=resnet_input_channels,
                    output_channels=output_channels,
                    resnet_groups=resnet_groups,
                    temb_channels=temb_channels,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=resnet_act_fn,
                    resnet_eps=resnet_eps,
                    dropout=dropout,
                )
            )

            self.attentions.append(
                Transformer2DModel(
                    num_layers=transformer_layers_per_block[i],
                    input_channels=output_channels,
                    output_channels=output_channels,
                    attention_head_dim=output_channels // num_attention_heads,
                    num_attention_heads=num_attention_heads,
                    cross_attention_dim=cross_attention_dim,
                    resnet_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.resnets = nn.ModuleList(self.resnets)
        self.attentions = nn.ModuleList(self.attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(output_channels, output_channels)])
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
    def __init__(
        self,
        *,
        num_layers: int,
        input_channels: int,
        output_channels: int,
        temb_channels: int,
        prev_output_channel: int,
        add_upsample: bool,
        resnet_act_fn: Callable,
        resnet_eps: float,
        resnet_groups: int,
        resnet_time_scale_shift: str,
        dropout: float,
        **kwargs,
    ):
        super().__init__()
        self.has_cross_attention = False
        self.resnets = []

        for i in range(num_layers):
            res_skip_channels = input_channels if (i == num_layers - 1) else output_channels
            resnet_input_channels = prev_output_channel if i == 0 else output_channels
            resnet_input_channels = res_skip_channels + resnet_input_channels

            self.resnets.append(
                ResnetBlock2D(
                    input_channels=resnet_input_channels,
                    output_channels=output_channels,
                    resnet_groups=resnet_groups,
                    temb_channels=temb_channels,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=resnet_act_fn,
                    resnet_eps=resnet_eps,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(self.resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(output_channels, output_channels)])
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
