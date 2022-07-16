import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
from common import exec_color, exec_edge_color, exec_fullname


script_dir = os.path.dirname(__file__)
exp_name = 'exp_tuning_time'
out_fname = os.path.join(script_dir, 'pdfs', f'{exp_name}.pdf')

# plt.style.use('ggplot')

font = {'family': 'serif', 'serif': ['Gentium Basic'], 'size': 15}
plt.rc('font', **font)

# plt.rcParams['text.color'] = 'blue'
# plt.rc('text', **{'color': 'black'})

executors = ['(A) OnnxRuntime', '(B) AutoTVM', '(C) Ansor', '(D) TensorRT', '(E) Hidet (ours)']
models = ['ResNet50', 'Inception V3', 'MobileNet V2', 'Bert', 'GPT-2']

# data[i, j] = network i on executor j

colors = [
    [
        # '#6DB1FF',
        # '#00C2A8',
        '#54C45E',
        # '#FFE342',
        '#FF8F8F',
        '#FC9432',
    ],
    [
        '#107115',
        '#008573',
        '#008A0E',
        '#FCCE14',
        '#CC4E00',
    ]
]

y_min = 0
y_max = 19 * 60

data = {
    'hidet': [20, 45, 22, 5, 5,
              97 / 5
              ],
    'ansor': [210, 516, 228, 51, 52,
              1057 / 5
              ],
    'autotvm': [480, 900, 558,
                2.3, 2.2,
                # 16, 16,  # 2.3 and 2.2 are smaller than y_min, use 16 here for visualization and label 2m in the figure.
                1942 / 5
                ]
}
tick_labels = [
    'ResNet50',
    'InceptionV3',
    'MbNetV2',
    'Bert',
    'GPT-2',
    'Average'
]


def minutes_to_text(v):
    if v < 60:
        return '{:.0f}m'.format(v)
        # return '{:.1f}h'.format(v / 60)
    else:
        return '{:.0f}h'.format(v / 60)


def main():
    fig: plt.Figure = plt.figure(
        figsize=(7.3, 3.2),
        tight_layout=True
    )
    bar_sep_width = 0.1
    bar_width = 0.3
    sep_width = 0.4
    num_inputs = len(data['hidet'])
    ax: plt.Axes = fig.add_subplot()
    executors = ['autotvm', 'ansor', 'hidet']
    bars = []
    bar_labels = []
    for idx, executor in enumerate(executors):
        x = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + idx * (bar_width + bar_sep_width)
        tick_label = [minutes_to_text(v) for v in data[executor]]
        # if executor == 'AutoTVM':
        #     tick_label[-3:-1] = ['2m', '2m']
            # tick_label = [minutes_to_text(v) for v in data[executor][:-2]] + ['2m', '2m']
        bar_labels.append(tick_label)
        bar = ax.bar(x, data[executor], color=exec_color[executor], tick_label=tick_label, edgecolor=exec_edge_color[executor], width=bar_width, label=exec_fullname[executor])
        bars.append(bar)

    xticks = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + (len(executors) - 1) * (bar_width + bar_sep_width) / 2.0
    ax.set_xticks(xticks)
    ax.set_xticklabels(tick_labels,
                       # rotation=-10,
                       ha='center',
                       # fontsize='small'
                       )
    ax.set_xlim(left=-bar_width, right=max(xticks) + (bar_width + bar_sep_width) + bar_width)
    ax.set_ylabel('Tuning Cost (Hours)')
    # ax.set_yscale('log', base=1.1)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_yticks([60, 6 * 60, 12 * 60, 18 * 60])
    ax.set_yticklabels(['1 h', '6 h', '12 h', '18 h'])
    # ax.set_xlabel('')
    fmt = '%.0f'
    for i, bar in enumerate(bars):
        ax.bar_label(bar,
                     labels=bar_labels[i],
                     # fmt=fmt,
                     label_type='edge', padding=2,
                     # fontsize='x-small'
                     )

    lgd = ax.legend(
        ncol=3,
        # loc='upper center',
        # bbox_to_anchor=(0.46, 1)
    )

    def bold(s):
        return r'${' + s + r'}$'

    ax.text(3.5, 8 * 60, '{} speedup tuning by \n'.format(exec_fullname['hidet']) +
                         '{}x (AutoTVM) and {}x (Ansor)\n'.format(bold('20'), bold('11')) +
                         'on average.'
            )

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(out_fname,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


if __name__ == '__main__':
    main()
