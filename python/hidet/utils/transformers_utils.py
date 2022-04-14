import sys
import os
import subprocess


def export_transformer_model_as_onnx(model_name='bert-base-uncased', feature='default', output_dir='./', save_name=None, skip_exists=True):
    """
    Export a model from transformers package.

    Parameters
    ----------
    model_name: str
        The model name.
    feature: str
        The feature of the exported model.
    output_dir: str
        The output dir.
    save_name: str
        The model file name.
    skip_exists: bool
        Skip export if target exists. Default True.

    Returns
    -------
    path: str
        The path to the exported onnx model.

    Examples
    --------
    Call export_transformer_model_as_onnx() will download (when needed) the requested model and export it to an onnx model.
    The function will return './bert-base-uncased.onnx', which can be load by onnx package.
    """
    if save_name is None:
        save_name = '{}.onnx'.format(model_name)
    target_path = os.path.join(output_dir, save_name)
    if os.path.exists(target_path):
        return target_path
    command = '{} -m transformers.onnx --model {} --feature {} {}'.format(sys.executable, model_name, feature, output_dir)
    print("Running '{}'".format(command))
    subprocess.run(command.split(), check=True)
    os.rename(os.path.join(output_dir, 'model.onnx'), target_path)
    return target_path


if __name__ == '__main__':
    export_transformer_model_as_onnx()
