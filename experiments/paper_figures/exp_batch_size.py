import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
from common import exec_color, exec_edge_color, exec_fullname, batch_data, batch_sizes, hline_alpha, hline_color


script_dir = os.path.dirname(__file__)
exp_name = 'exp_batch_size'
out_fname = os.path.join(script_dir, 'pdfs', f'{exp_name}.pdf')

# plt.style.use('ggplot')

font = {'family': 'serif', 'serif': ['Gentium Basic'], 'size': 15}
plt.rc('font', **font)

# plt.rcParams['text.color'] = 'blue'
# plt.rc('text', **{'color': 'black'})

executors = ['(A) OnnxRuntime', '(B) AutoTVM', '(C) Ansor', '(D) TensorRT', '(E) Hidet (ours)']
models = ['ResNet50', 'Inception V3', 'MobileNet V2', 'Bert', 'GPT-2']

# data[i, j] = network i on executor j


y_min = 0
# y_max = 19 * 60

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
        figsize=(6.5, 2.8),
        tight_layout=True
    )
    bar_sep_width = 0.015
    bar_width = 0.08
    sep_width = 0.2
    num_inputs = len(batch_data['hidet'])
    ax: plt.Axes = fig.add_subplot()
    executors = ['torch', 'ort', 'autotvm', 'ansor', 'hidet']
    bars = []
    # bar_labels = []
    for idx, executor in enumerate(executors):
        x = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + idx * (bar_width + bar_sep_width)
        # tick_label = [minutes_to_text(v) for v in batch_data[executor]]
        # if executor == 'AutoTVM':
        #     tick_label[-3:-1] = ['2m', '2m']
            # tick_label = [minutes_to_text(v) for v in data[executor][:-2]] + ['2m', '2m']
        # bar_labels.append(tick_label)
        # print(x)
        bar = ax.bar(x, batch_data[executor], color=exec_color[executor], edgecolor=exec_edge_color[executor], width=bar_width, label=exec_fullname[executor])
        bars.append(bar)
    for idx, bs in enumerate(batch_sizes):
        y = batch_data['hidet'][idx]
        x_min_all = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + 0 * (bar_width + bar_sep_width)
        x_max_all = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + (len(executors) - 1) * (bar_width + bar_sep_width)
        x_min = x_min_all[idx] - bar_width * 0.5
        x_max = x_max_all[idx] - bar_width * 0.5
        # print(x_min_all)
        # print(x_max_all)
        # print(x_min, x_max)
        # ax.axhline(y=y, xmin=x_min, xmax=x_max, alpha=hline_alpha, color=hline_color, linestyle='dashed', lw=1.0)
        # ax.axline(xy1=(x_min, y), xy2=(x_max, y), # y=y, xmin=x_min, xmax=x_max,
        #           alpha=hline_alpha, color=hline_color, linestyle='dashed', lw=1.0)
        ax.hlines(y=y, xmin=x_min, xmax=x_max, alpha=hline_alpha, color=hline_color, linestyle='dashed', lw=1.0)
        # ax.annotate('l', (x_min, y))
        # ax.annotate('r', (x_max, y))

        # if idx == 0:
        #     break

    xticks = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + (len(executors) - 1) * (bar_width + bar_sep_width) / 2.0
    ax.set_xticks(xticks)
    xticklabels = [str(v) for v in batch_sizes]
    ax.set_xticklabels(xticklabels,
                       # rotation=-10,
                       ha='center',
                       # fontsize='small'
                       )
    ax.set_xlim(left=-bar_width, right=max(xticks) + (bar_width + bar_sep_width) * 2 + bar_width)
    ax.set_ylabel('Latency (ms)')
    # ax.set_yscale('log', base=1.1)
    y_min = 0
    y_max = 12.5
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_yticks([0, 5, 10])
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
        ncol=3,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.028)
    )

    # def bold(s):
    #     return r'${' + s + r'}$'

    # ax.text(3.5, 8 * 60, '{} speedup tuning by \n'.format(exec_fullname['hidet']) +
    #                      '{}x (AutoTVM) and {}x (Ansor)\n'.format(bold('20'), bold('11')) +
    #                      'on average.'
    #         )

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(out_fname,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


if __name__ == '__main__':
    main()
