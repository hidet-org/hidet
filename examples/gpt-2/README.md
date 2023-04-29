# GPT-2 Demo

This example demonstrates how to use Hidet to compile and run a GPT-2 model. 

## Requirements

This example requires a nightly build version of hidet before we release the next version (v0.2.4) to PyPI:
Run the following commands under the `examples/gpt-2` directory to install the required packages:
```console
$ pip install --pre --extra-index-url https://download.hidet.org/whl hidet
$ pip install -r requirements.txt
```

## Usage

```bash
$ python main.py
>>> Alan Turing theorized that computers would one day become
generating: 100%|██████████████████████████████| 30/30 [00:00<00:00, 128.30it/s]
Alan Turing theorized that computers would one day become the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations 
```

## Configs
```bash
Usage: main.py [OPTIONS]

Options:
  --max-num-tokens INTEGER        Max number of total tokens to process and
                                  generate  [default: 40]
  --use-fp16                      Use fp16
  --model-size [124M|355M|774M|1558M]
                                  [default: 124M]
  --tune                          Tune the operators for better performance.
                                  May take several minutes.
  --help                          Show this message and exit.
```

## Acknowledgements
We referred to the [picoGPT](https://github.com/jaymody/picoGPT)'s clean and simple implementation of GPT-2.
