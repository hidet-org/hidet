import argparse
import datetime
import subprocess
import hidet


parser = argparse.ArgumentParser('Benchmark hidet performance.')
parser.add_argument('--git-commit', default=None, type=str, help='Git commit hash.')
parser.add_argument('--space', default=2, type=int, help='Search space of hidet.')
parser.add_argument('--report', default='./report.txt', type=str, help='Report file path.')


def info(args) -> str:
    envs = [
        '# {}'.format(datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y-%m-%d')),
        '- Hidet version: {}'.format(hidet.__version__),
        '- Git commit: {}'.format(args.git_commit),
        '',
    ]
    return '\n'.join(envs)


def main():
    args = parser.parse_args()
    commands = [
        f'hidet bench --space {args.space} --dtype float32 --report resnet50_f32.txt --tensor-core resnet --models resnet50',
        f'hidet bench --space {args.space} --dtype float16 --report resnet50_f16.txt --tensor-core resnet --models resnet50',
        f'hidet bench --space {args.space} --dtype float32 --report bert-seq128-f32.txt --tensor-core nlp --seq-length 128 --models bert-base-uncased',
        f'hidet bench --space {args.space} --dtype float16 --report bert-seq128-f16.txt --tensor-core nlp --seq-length 128 --models bert-base-uncased',
    ]
    with open(args.report, 'w') as f:
        f.write(info(args))
        for command in commands:
            output_file = command.split('--report ')[1].split(' ')[0]
            subprocess.run(command.split(), check=True)
            with open(output_file, 'r') as g:
                f.write('---\n')
                f.write('Command:\n`{}`\n'.format(command))
                f.write(g.read())
                f.write('\n')


if __name__ == '__main__':
    main()
