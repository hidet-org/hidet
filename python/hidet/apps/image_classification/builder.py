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

from transformers import PretrainedConfig

from hidet.apps import hf
from hidet.apps.image_classification.processing.image_processor import BaseImageProcessor
from hidet.apps.image_classification.app import ImageClassificationApp
from hidet.apps.image_classification.modeling.pretrained import PretrainedModelForImageClassification
from hidet.apps.modeling_outputs import ImageClassifierOutput
from hidet.graph import trace_from
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.tensor import Tensor, symbol
from hidet.runtime.compiled_app import create_compiled_app

import hidet


def create_image_classifier(
    name: str,
    revision: Optional[str] = None,
    dtype: str = "float32",
    device: str = "cuda",
    batch_size: int = 1,
    kernel_search_space: int = 2,
):
    # load the huggingface config according to (model, revision) pair
    config: PretrainedConfig = hf.load_pretrained_config(name, revision=revision)

    # load model instance by architecture, assume only 1 architecture for now
    model = PretrainedModelForImageClassification.create_pretrained_model(
        config, revision=revision, dtype=dtype, device=device
    )
    inputs: Tensor = symbol([batch_size, 3, 224, 224], dtype=dtype, device=device)
    outputs: ImageClassifierOutput = model.forward(inputs)
    graph: FlowGraph = trace_from(outputs.logits, inputs)

    graph = hidet.graph.optimize(graph)

    compiled_graph = graph.build(space=kernel_search_space)

    return ImageClassificationApp(
        compiled_app=create_compiled_app(
            graphs={"image_classifier": compiled_graph}, modules={}, tensors={}, attributes={}, name=name
        )
    )


def create_image_processor(name: str, revision: Optional[str] = None, **kwargs):
    # load the huggingface config according to (model, revision) pair
    config: PretrainedConfig = hf.load_pretrained_config(name, revision=revision)

    processor = BaseImageProcessor.load_module(config.architectures[0])

    return processor(**kwargs)
