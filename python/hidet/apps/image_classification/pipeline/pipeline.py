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
from typing import Any, Iterable, Optional, Sequence
from hidet.apps import hf
from hidet.apps.image_classification.builder import create_image_classifier, create_image_processor
from hidet.apps.image_classification.processing import BaseImageProcessor, ChannelDimension, ImageInput
from hidet.graph.tensor import Tensor


class ImageClassificationPipeline:
    def __init__(
        self,
        name: str,
        revision: Optional[str] = None,
        batch_size: int = 1,
        pre_processor: Optional[BaseImageProcessor] = None,
        dtype: str = "float32",
        device: str = "cuda",
        kernel_search_space: int = 2,
    ):
        if pre_processor is None:
            self.pre_processor = create_image_processor(name, revision)
        else:
            self.pre_processor = pre_processor

        self.model = create_image_classifier(name, revision, dtype, device, batch_size, kernel_search_space)
        self.config = hf.load_pretrained_config(name, revision)

    def __call__(self, model_inputs: Any, **kwargs):
        """
        Run through image classification pipeline end to end.
        images: ImageInput
            List or single instance of numpy array, PIL image, or torch tensor
        input_data_format: ChannelDimension
            Input data is channel first or last
        batch_size: int (default 1)
            Batch size to feed model inputs
        top_k: int (default 5)
            Return scores for top k results
        """
        if not isinstance(model_inputs, Iterable):
            model_inputs = [model_inputs]
        if not isinstance(model_inputs, Sequence):
            model_inputs = list(model_inputs)

        assert isinstance(model_inputs, Sequence)

        processed_inputs = self.preprocess(model_inputs, **kwargs)
        model_outputs = self.forward(processed_inputs, **kwargs)
        outputs = self.postprocess(model_outputs, **kwargs)

        return outputs

    def preprocess(self, images: ImageInput, input_data_format: ChannelDimension, **kwargs):
        # TODO accept inputs other than ImageInput type, e.g. url or dataset
        return self.pre_processor(images, input_data_format=input_data_format, **kwargs)

    def postprocess(self, model_outputs: Tensor, top_k: int = 5, **kwargs):
        top_k = min(top_k, self.config.num_labels)
        torch_outputs = model_outputs.torch()
        values, indices = torch_outputs.topk(top_k, sorted=False)
        labels = [[self.config.id2label[int(x.item())] for x in t] for t in indices]
        return [[{"label": label, "score": value.item()} for label, value in zip(a, b)] for a, b in zip(labels, values)]

    def forward(self, model_inputs: Tensor, **kwargs) -> Tensor:
        return self.model.classify([model_inputs])[0]
