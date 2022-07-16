import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
from common import exec_color, exec_edge_color, exec_fullname


script_dir = os.path.dirname(__file__)
exp_name = 'exp_input_sensitivity'
out_fname = os.path.join(script_dir, 'pdfs', f'{exp_name}.pdf')

# plt.style.use('ggplot')

font = {'family': 'serif', 'serif': ['Gentium Basic'], 'size': 15}
plt.rc('font', **font)

# plt.rcParams['text.color'] = 'blue'
# plt.rc('text', **{'color': 'black'})

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

data = {
    'hidet': [1.308, 1.353, 1.351, 1.357, 1.350, 1.353, 1.343, 1.346],
    'ansor': [2.057, 2.296, 1.856, 6.534, 1.999, 3.425, 27.357, float('NaN')],
    # 'AutoTVM': [2.141, 2.728, 1.941, 13.225, 2.422, 4.302, 37.949, 354.9]
    'autotvm': [2.141, 2.728, 1.941, 13.225, 2.422, 4.302, 37.949, float('NaN')]
}
tick_labels = [2048, 2047, 2046, 2045, 2044, 2043, 2042, 2039]

y_max = 6.0


def main():
    fig: plt.Figure = plt.figure(
        figsize=(6.5, 3.2),
        tight_layout=True
    )
    bar_sep_width = 0.05
    bar_width = 0.3
    sep_width = 0.3
    num_inputs = len(data['hidet'])
    ax: plt.Axes = fig.add_subplot()
    executors = ['autotvm', 'ansor', 'hidet']
    bars = []
    for idx, executor in enumerate(executors):
        x = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + idx * (bar_width + bar_sep_width)
        bar = ax.bar(x, data[executor], color=exec_color[executor], #colors[0][idx],
                     edgecolor=exec_edge_color[executor], width=bar_width, label=exec_fullname[executor])
        bars.append(bar)
        for i, y in enumerate(data[executor]):
            if y >= y_max:
                xy = (x[i], y_max)
                xx = x[i]
                yy = y_max * 0.8
                if idx == 0:
                    if np.isnan(data['ansor'][i]):
                        xx += bar_width * 0.8
                    else:
                        xx -= bar_width * 1.8
                elif idx == 1:
                    xx += bar_width * 1.0
                xytext = (xx, yy)
                ax.annotate('%.0f' % y, xy, xytext=xytext,
                            fontsize='small',
                            arrowprops=dict(
                                arrowstyle='->',
                                connectionstyle="arc3,rad=-0.05",
                                color="k",
                                # shrinkA=5,
                                # shrinkB=5,
                                # patchB=l,
                            )
                            )
            if np.isnan(y) and executor == 'autotvm':
                xy = (x[i], 0.0)
                xytext = (x[i] - bar_width * 0.55, y_max * 0.28)
                ax.annotate('Failed', xy, xytext=xytext,
                            fontsize='small',
                            arrowprops=dict(
                                arrowstyle='->',
                                # connectionstyle="arc3,rad=-0.05",
                                connectionstyle="arc3,rad=+0.1",
                                color="k",
                                # shrinkA=5,
                                # shrinkB=5,
                                # patchB=l,
                            ))
                xy = (x[i] + bar_width + bar_sep_width, 0.0)
                ax.annotate('Failed', xy, xytext=xytext,
                            fontsize='small',
                            arrowprops=dict(
                                arrowstyle='->',
                                # connectionstyle="arc3,rad=-0.05",
                                connectionstyle="arc3,rad=-0.1",
                                color="k",
                                # shrinkA=5,
                                # shrinkB=5,
                                # patchB=l,
                            ))

    xticks = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + (len(executors) - 1) * (bar_width + bar_sep_width) / 2.0
    ax.set_xticks(xticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(left=-bar_width, right=max(xticks) + (bar_width + bar_sep_width) + bar_width)
    ax.set_ylim(bottom=0, top=y_max)
    ax.set_ylabel('Latency (ms)')
    ax.set_xlabel('Matrix Multiplication (M=N=K)')
    fmt = '%.1f'
    # for bar in bars:
    #     ax.bar_label(bar, fmt=fmt, label_type='edge', padding=2, fontsize='x-small')

    ax.legend()

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(out_fname,
                # bbox_extra_artists=(lgd,),
                bbox_inches='tight')


if __name__ == '__main__':
    main()
