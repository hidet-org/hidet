from typing import List
import operator
import math
import functools
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
from common import end2end_data, exec_color, exec_edge_color, exec_fullname

script_dir = os.path.dirname(__file__)
exp_name = 'exp_trt'
out_fname = os.path.join(script_dir, 'pdfs', f'{exp_name}.pdf')

# plt.style.use('ggplot')

font = {'family': 'serif', 'serif': ['Gentium Basic'], 'size': 15}
plt.rc('font', **font)

# plt.rcParams['text.color'] = 'blue'
# plt.rc('text', **{'color': 'black'})

executor_name = {
    'torch': 'PyTorch',
    'ort': 'OnnxRuntime',
    'autotvm': 'AutoTVM',
    'ansor': 'Ansor',
    'trt': 'TensorRT',
    'hidet': 'Hidet'
}
executors = [
    # 'torch',
    # 'ort',
    # 'autotvm',
    # 'ansor',
    'trt',
    'hidet',
]
models = ['ResNet50', 'IncpV3', 'MbNetV2', 'Bert', 'GPT-2', 'Geo-Mean']

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
#     'trt': '#76B900',
#     'hidet': '#FC9432',
#     # 'trt': '#E81313',
# }

data = end2end_data


# add geo mean
def geo_mean(numbers: List[float]):
    numbers = list(filter(lambda v: v > 0.0, numbers))
    product = functools.reduce(operator.mul, numbers, 1.0)
    return product ** (1.0 / len(numbers))


for name in data:
    data[name].append(geo_mean(data[name]))


use_geo_mean = True


def main():
    fig: plt.Figure = plt.figure(
        figsize=(6.4, 2.8),
        tight_layout=True
    )
    bar_sep_width = 0.05
    bar_width = 0.2
    sep_width = 0.3
    num_inputs = len(data['hidet'])
    ax: plt.Axes = fig.add_subplot()
    bars = []
    bar_labels = []
    for idx, executor in enumerate(executors):
        x = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + idx * (bar_width + bar_sep_width)
        # tick_label = [minutes_to_text(v) for v in data[executor]]
        # if executor == 'AutoTVM':
        #     tick_label[-3:-1] = ['2m', '2m']
        # tick_label = [minutes_to_text(v) for v in data[executor][:-2]] + ['2m', '2m']
        # bar_labels.append(tick_label)
        bar = ax.bar(x, data[executor], color=exec_color[executor], edgecolor=exec_edge_color[executor], width=bar_width, label=exec_fullname[executor])
        bars.append(bar)

    xticks = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + (len(executors) - 1) * (bar_width + bar_sep_width) / 2.0
    ax.set_xticks(xticks)
    ax.set_xticklabels(models,
                       # rotation=-10,
                       ha='center',
                       # fontsize='small'
                       )
    ax.set_xlim(left=-bar_width, right=max(xticks) + (bar_width + bar_sep_width) + bar_width)
    ax.set_ylabel('Latency (ms)')
    # ax.set_yscale('log', base=1.1)
    ax.set_ylim(bottom=0, top=4.0)
    # ax.set_yticks([60, 6 * 60, 12 * 60, 18 * 60])
    # ax.set_yticklabels(['1 h', '6 h', '12 h', '18 h'])
    # ax.set_xlabel('')
    # fmt = '%.0f'
    # for i, bar in enumerate(bars):
    #     ax.bar_label(bar,
    #                  labels=bar_labels[i],
    #                  # fmt=fmt,
    #                  label_type='edge', padding=2,
    #                  # fontsize='x-small'
    #                  )

    lgd = ax.legend(
        ncol=2,
        # loc='upper center',
        # bbox_to_anchor=(0.46, 1)
    )

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(out_fname,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


if __name__ == '__main__':
    main()
