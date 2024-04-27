## Hidet Image Classification Compiled App

### Quickstart

```
from hidet.apps.image_classification.pipeline.pipeline import ImageClassificationPipeline
from hidet.apps.image_classification.processing.image_processor import ChannelDimension
from datasets import load_dataset


dataset = load_dataset("huggingface/cats-image", split="test", trust_remote_code=True)

pipeline = ImageClassificationPipeline("microsoft/resnet-50", batch_size=1, kernel_search_space=0)

res = pipeline(dataset["image"], input_data_format=ChannelDimension.CHANNEL_LAST, top_k=3)
```

An image classifier app currently only supports ResNet50 from Huggingface. Currently supports PIL + torch/hidet tensors as image input. 

Load sample datasets using the datasets library, and change label ids back to string labels using the Huggingface config. Returns the top k candidates with the highest score.

If the weights used are not public, be sure to modify `hidet.toml` so that option `auth_tokens.for_huggingface` is set to your Huggingface account credential.

### Model Details

A `PretrainedModelForImageClassification` is a `PretrainedModel` that allows us to `create_pretrained_model` from a Huggingface identifier. `PretrainedModelForImageClassification` defines a forward function that accepts Hidet tensors as input and returns logits as output.

Interact with a `PretrainedModelForImageClassification` using `ImageClassificationPipeline`. The pipeline instantiates a pre-processor that adapts the image type for Hidet and performs transformations on the image before calling the pretrained model graph. Specify batch size and model name using the pipeline constructor.

