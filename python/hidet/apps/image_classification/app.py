from typing import Sequence

from hidet.graph.tensor import Tensor
from hidet.runtime.compiled_app import CompiledApp


class ImageClassificationApp:
    def __init__(self, compiled_app: CompiledApp):
        super().__init__()
        self.compiled_app: CompiledApp = compiled_app

    def classify(self, input_images: Sequence[Tensor]):
        return self.compiled_app.graphs["image_classifier"].run_async(input_images)
