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
import os
from hidet.utils import hidet_cache_file


def export_torchvision_model_as_onnx(model_name: str, output_path: str, skip_existed: bool = True, precision='float32'):
    assert precision == 'float32'
    if skip_existed and os.path.exists(output_path):
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import torchvision
    import torch

    if model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True).cuda()
        input_shape = [1, 3, 224, 224]
    elif model_name == 'inception_v3':
        model = torchvision.models.inception_v3(pretrained=True, transform_input=False, aux_logits=True).cuda()
        input_shape = [1, 3, 299, 299]
    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True).cuda()
        input_shape = [1, 3, 224, 224]
    elif model_name == 'resnext50':
        model = torchvision.models.resnext50_32x4d(pretrained=True).cuda()
        input_shape = [1, 3, 224, 224]
    else:
        raise NotImplementedError(model_name)

    model.eval()
    dummy_input = torch.randn(*input_shape, device='cuda')
    input_names = ['data']
    output_names = ['output']
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=output_path,
        training=torch.onnx.TrainingMode.PRESERVE,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=False,
        dynamic_axes={'data': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )


if __name__ == '__main__':
    names = ['resnet50', 'inception_v3', 'mobilenet_v2']
    for name in names:
        export_torchvision_model_as_onnx(name, hidet_cache_file('onnx', f'{name}.onnx'))
