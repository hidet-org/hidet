import sys
import subprocess


def export_transformer_model_as_onnx(model_name='bert-base-uncased', feature='default', output_dir='./'):
    command = '{} -m transformers.onnx --model {} --feature {} {}'.format(sys.executable, model_name, feature, output_dir)
    print("Running '{}'".format(command))
    subprocess.run(command.split(), check=True)
