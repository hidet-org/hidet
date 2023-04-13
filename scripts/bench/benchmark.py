import os
import argparse
import datetime
import subprocess
import time
import pytz
import distro
from cuda import cudart
import torch
import hidet

hidet.option.cache_dir(os.path.join(hidet.option.get_cache_dir(), 'benchmark'))
cache_dir = hidet.option.get_cache_dir()


parser = argparse.ArgumentParser('Benchmark hidet performance.')
parser.add_argument('--git-prev-commit', default=None, type=str, help='Previous git commit hash.')
parser.add_argument('--git-commit', type=str, help='Git commit hash.')
parser.add_argument('--keep-cache', default=False, action='store_true', help='Keep cache.')
parser.add_argument('--space', default=2, type=int, help='Search space of hidet.')
parser.add_argument('--report', default='./report.txt', type=str, help='Report file path.')


def nvidia_gpu_driver() -> str:
    with open('/proc/driver/nvidia/version', 'r') as f:
        return f.readline().split('Module')[1].split()[0]


def nvidia_cuda_version() -> str:
    v = cudart.cudaRuntimeGetVersion()[1]
    return '{}.{}'.format(v // 1000, v % 1000 // 10)


def info(args) -> str:
    envs = [
        '# {}'.format(datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')),
        '- Hidet version: {}'.format(hidet.__version__),
        '- PyTorch version: {}'.format(torch.__version__),
        '- OS: {}'.format(distro.name(pretty=True)),
        '- GPU: {}'.format(cudart.cudaGetDeviceProperties(0)[1].name.decode('utf-8')),
        '- GPU driver: {} ({})'.format(nvidia_gpu_driver(), nvidia_cuda_version()),
    ]
    if args.git_prev_commit and args.git_commit:
        envs += [
            '- Git diff: {}'.format(args.git_prev_commit + '...' + args.git_commit),
        ]
    else:
        envs += [
            '- Git commit: {}'.format(args.git_commit),
        ]
    envs.append('')
    return '\n'.join(envs)


def main():
    args = parser.parse_args()

    if not args.keep_cache:
        print('Clearing hidet operator cache...')
        hidet.utils.hidet_clear_op_cache()

    commands = [
        f'hidet bench --cache-dir {cache_dir} --space {args.space} --dtype float32 --report resnet50_f32.txt --tensor-core resnet --models resnet50',
        f'hidet bench --cache-dir {cache_dir} --space {args.space} --dtype float16 --report resnet50_f16.txt --tensor-core resnet --models resnet50',
        f'hidet bench --cache-dir {cache_dir} --space {args.space} --dtype float32 --report bert-seq128-f32.txt --tensor-core nlp --seq-length 128 --models bert-base-uncased',
        f'hidet bench --cache-dir {cache_dir} --space {args.space} --dtype float16 --report bert-seq128-f16.txt --tensor-core nlp --seq-length 128 --models bert-base-uncased',
        # f'hidet bench --cache-dir {cache_dir} --space {args.space} --dtype float32 --report gpt2-seq128-f32.txt --tensor-core nlp --seq-length 128 --models gpt2',
        # f'hidet bench --cache-dir {cache_dir} --space {args.space} --dtype float16 --report gpt2-seq128-f16.txt --tensor-core nlp --seq-length 128 --models gpt2',
    ]
    with open(args.report, 'w') as f:
        t1 = time.time()
        f.write(info(args) + '\n')
        for idx, command in enumerate(commands):
            output_file = command.split('--report ')[1].split(' ')[0]
            subprocess.run(command.split(), check=True)
            with open(output_file, 'r') as g:
                if idx == 0:
                    f.write(g.read())
                else:
                    f.write(g.readlines()[-1])
        t2 = time.time()
        f.write('\n')
        f.write('Time: {:.2f} hours\n'.format((t2 - t1) / 60 / 60))


if __name__ == '__main__':
    main()
