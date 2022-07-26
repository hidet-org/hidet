import sys
import os
import subprocess


def export_transformer_model_as_onnx(model_name: str, output_path: str, precision='float32', feature='default', skip_exists=True):
    """
    Export a model from transformers package.

    Parameters
    ----------
    model_name: str
        The model name.
    output_path: str
        The output path.
    feature: str
        The feature of the exported model.
    skip_exists: bool
        Skip export if target exists. Default True.

    Examples
    --------
    Call export_transformer_model_as_onnx() will download (when needed) the requested model and export it to an onnx model.
    The function will return '{output_dir}/bert-base-uncased.onnx', which can be load by onnx package.
    """
    assert precision == 'float32'
    if skip_exists and os.path.exists(output_path):
        return
    temp_dir = '/tmp/hidet'
    command = '{} -m transformers.onnx --model {} --feature {} {}'.format(sys.executable, model_name, feature, temp_dir)
    print("Running '{}'".format(command))
    subprocess.run(command.split(), check=True)
    os.rename(os.path.join(temp_dir, 'model.onnx'), output_path)
    print('Model saved at: {}'.format(output_path))


if __name__ == '__main__':
    export_transformer_model_as_onnx(model_name='bert-base-uncased', output_path='./bert.onnx')
