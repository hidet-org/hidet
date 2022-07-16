from typing import List
import operator
from copy import copy
import math
import functools
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
from common import end2end_data, exec_color, exec_edge_color, hline_color, hline_alpha, exec_fullname

script_dir = os.path.dirname(__file__)
exp_name = 'exp_end2end'
out_fname = os.path.join(script_dir, 'pdfs', f'{exp_name}.pdf')

# plt.style.use('ggplot')

font = {'family': 'serif', 'serif': ['Gentium Basic'], 'size': 15}
plt.rc('font', **font)

# plt.rcParams['text.color'] = 'blue'
# plt.rc('text', **{'color': 'black'})

# executor_name = {
#     'torch': 'PyTorch',
#     'ort': 'OnnxRuntime',
#     'autotvm': 'AutoTVM',
#     'ansor': 'Ansor',
#     'trt': 'TensorRT',
#     'hidet': 'Hidet (ours)'
# }
executor_name = copy(exec_fullname)
executor_name['hidet'] += ' (ours)'
executors = [
    'torch',
    'ort',
    'autotvm',
    'ansor',
    # 'trt',
    'hidet',
]
models = ['ResNet50', 'Inception V3', 'MobileNet V2', 'Bert', 'GPT-2', 'Geo-Mean']

colors = [
    [
        '#6DB1FF',
        '#00C2A8',
        '#54C45E',
        # '#FFE342',
        '#FF8F8F',
        '#FC9432',
    ],
    [
    ]
]
deep_colors = [
    '#107115',
    '#008573',
    '#008A0E',
    '#FCCE14',
    '#CC4E00',
]

# color = {
#     'torch': '#6DB1FF',
#     'ort': '#00C2A8',
#     'autotvm': '#54C45E',
#     'ansor': '#FF8F8F',
#     'trt': '#FF80DF',
#     'hidet': '#FC9432',
#     # 'hidet': '#E81313',
# }

data = end2end_data


# add geo mean
def geo_mean(numbers: List[float]):
    numbers = list(filter(lambda v: v > 0.0, numbers))
    product = functools.reduce(operator.mul, numbers, 1.0)
    return product ** (1.0 / len(numbers))


for name in data:
    data[name].append(geo_mean(data[name]))

y_max = [5, 9, 4.5, 6.0, 6.0, 6.0]

use_geo_mean = True


def main():
    fig: plt.Figure = plt.figure(
        figsize=(12.8, 2.8),
        tight_layout=True
    )
    axes = fig.subplots(1, len(models), sharex=True)
    big_ax: plt.Axes = fig.add_subplot(111, frameon=False)  # The big subplot
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    big_ax.set_ylabel('Latency (ms)')
    num_executors = len(executors)
    labels = [chr(ord('A') + i) for i in range(num_executors)]
    executor_colors = [exec_color[e] for e in executors]
    executor_edge_colors = [exec_edge_color[e] for e in executors]
    for j in range(len(models)):
        ax: plt.Axes = axes[j]
        model_name = models[j]
        latencies = [data[e][j] for e in executors]
        print(model_name, latencies)
        bars = ax.bar(range(num_executors), latencies, tick_label=labels, color=executor_colors, edgecolor=executor_edge_colors)
        ax.axhline(y=data['hidet'][j], xmin=0.05, xmax=0.8, alpha=hline_alpha, color=hline_color, linestyle='dashed', lw=1.0)
        ax.text(x=executors.index('hidet') - 0.6, y=data['hidet'][j] + y_max[j] * 0.03, s='{:.2f}x'.format(min(latencies[:-1]) / data['hidet'][j]), fontsize='x-small')
        ax.set_ylim(top=y_max[j])
        ax.yaxis.set_major_formatter('{x:.1f}')
        ax.set_xlabel(model_name)
        if model_name in ['Bert', 'GPT-2']:
            xy = (executors.index('autotvm'), y_max[j])
            xytext = (xy[0] - 0.36,
                      xy[1] - y_max[j] * 0.12)
            ax.annotate(
                '%.0f' % data['autotvm'][j],
                xy,
                xytext=xytext,
                # fontsize='small',
                # arrowprops=dict(
                #     arrowstyle='->',
                #     connectionstyle="arc3,rad=-0.05",
                #     color="k",
                #     # shrinkA=5,
                #     # shrinkB=5,
                #     # patchB=l,
                # )
            )
        if j == 0:
            legends = ['({}) {}'.format(label, executor_name[e]) for label, e in zip(labels, executors)]
            lgd = fig.legend(bars, legends, loc='upper center', ncol=num_executors, bbox_to_anchor=(0.5, 1.08))

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(out_fname, bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    main()
