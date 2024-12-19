# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
from transformers import AutoConfig, PretrainedConfig
from diffusers import StableDiffusionPipeline


def load_pretrained_config(model: str, revision: Optional[str] = None) -> PretrainedConfig:
    return AutoConfig.from_pretrained(model, revision=revision)


def load_diffusion_pipeline(name: str, revision: Optional[str] = None, device: str = "cuda") -> StableDiffusionPipeline:
    with torch.device(device):
        return StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=name, torch_dtype=torch.float32, revision=revision
        )
