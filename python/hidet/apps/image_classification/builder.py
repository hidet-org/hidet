from typing import Optional

from transformers import PretrainedConfig

from hidet.apps import hf
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
    kernel_search_space: int = 2,
):
    # load the huggingface config according to (model, revision) pair
    config: PretrainedConfig = hf.load_pretrained_config(name, revision=revision)

    # load model instance by architecture, assume only 1 architecture for now
    model = PretrainedModelForImageClassification.create_pretrained_model(
        config, revision=revision, dtype=dtype, device=device
    )
    inputs: Tensor = symbol(["bs", 3, 224, 224], dtype=dtype, device=device)
    outputs: ImageClassifierOutput = model.forward(inputs)
    graph: FlowGraph = trace_from(outputs.logits, inputs)

    graph = hidet.graph.optimize(graph)

    compiled_graph = graph.build(space=kernel_search_space)

    return ImageClassificationApp(
        compiled_app=create_compiled_app(
            graphs={"image_classifier": compiled_graph}, modules={}, tensors={}, attributes={}, name=name
        )
    )


# def create_image_processor(
#     name: str,
#     revision: Optional[str] = None,
#     **kwargs
# ) -> BaseProcessor:
#     # load the huggingface config according to (model, revision) pair
#     config: PretrainedConfig = hf.load_pretrained_config(name, revision=revision)

#     processor = BaseImageProcessor.load_module(config, module_type=ModuleType.PROCESSING)

#     return processor(**kwargs)
