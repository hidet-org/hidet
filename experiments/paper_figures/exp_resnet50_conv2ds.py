import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
from common import exec_color, exec_edge_color, exec_fullname


script_dir = os.path.dirname(__file__)
exp_name = 'exp_resnet50_conv2ds'
out_fname = os.path.join(script_dir, 'pdfs', f'{exp_name}.pdf')

# plt.style.use('ggplot')

font = {'family': 'serif', 'serif': ['Gentium Basic'], 'size': 15}
plt.rc('font', **font)

# plt.rcParams['text.color'] = 'blue'
# plt.rc('text', **{'color': 'black'})

models = ['ResNet50', 'Inception V3', 'MobileNet V2', 'Bert', 'GPT-2']
executors = ['OnnxRuntime', 'Ansor', 'Hidet']

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


# y_max = 6.0


def read_data():
    data_path = './data/exp_resnet50_conv2ds_data/records.txt'
    name_map = {
        'ansor': 'Ansor',
        'ort': 'OnnxRuntime',
        'trt': 'TensorRT',
        'hidet': 'Hidet'
    }
    data = {
    }
    item_lines = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            items = line.split()
            name = str(name_map[items[2]])
            latency = float(items[5])
            conv_id = int(items[1].split('_')[-1])
            item_lines.append([name, conv_id, latency])
    item_lines = sorted(item_lines, key=lambda items: (items[0], items[1]))

    for items in item_lines:
        name, conv_id, latency = items
        if name not in data:
            data[name] = []
        data[name].append(latency)
    flops = read_flops()
    for name in data:
        # lst = [2 * a / b / (10**9) for a, b in zip([tp[0] for tp in flops], data[name])]    # 2 flops for single fma
        lst = [b for a, b in zip([tp[0] for tp in flops], data[name])]    # 2 flops for single fma
        # lst = list(enumerate(lst))
        # lst = reversed(sorted(lst, key=lambda tp: flops[tp[0]][3]))
        # lst = [v for i, v in lst]
        data[name] = lst[11:]

    return data


def read_flops():
    conv2d_mnk_path = './data/exp_resnet50_conv2ds_data/conv2d_mnk.txt'
    ret = []
    with open(conv2d_mnk_path, 'r') as f:
        for line in f.readlines():
            m, n, k = [int(v) for v in line.split()]
            ret.append([m * n * k, m, n, k])
    return ret


def main():
    fig: plt.Figure = plt.figure(
        figsize=(6.5, 2.8),
        tight_layout=True
    )
    bar_sep_width = 0.125
    bar_width = 0.3
    sep_width = 0.5
    data = read_data()
    num_inputs = len(data['Hidet'])
    ax: plt.Axes = fig.add_subplot()
    bars = []
    name_rmap = {
        'Ansor': 'ansor',
        'OnnxRuntime': 'ort',
        'Hidet': 'hidet'
    }
    for idx, executor in enumerate(executors):
        x = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + idx * (bar_width + bar_sep_width)
        y = data[executor]
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        bar = ax.bar(x, y, color=exec_color[name_rmap[executor]],
                     edgecolor=exec_edge_color[name_rmap[executor]],
                     width=bar_width, label=exec_fullname[name_rmap[executor]])
        # ax.plot(x, p(x), '-', color=colors[0][idx])
        bars.append(bar)

    xticks = np.arange(num_inputs) * (bar_width * len(executors) + sep_width) + (len(executors) - 1) * (bar_width + bar_sep_width) / 2.0
    ax.set_xticks([])
    # tick_labels = list(range(1, len(xticks) + 1))
    # ax.set_xticklabels(tick_labels)
    ax.set_xlim(left=-bar_width, right=max(xticks) + (bar_width + bar_sep_width) + bar_width)
    ax.set_ylim(bottom=0, top=0.2)
    ax.set_ylabel('Latency (ms)')
    ax.set_xlabel('Conv2d-Bn-ReLU in ResNet50')
    ax.set_yticks([0.0, 0.1, 0.2])
    fmt = '%.1f'
    # for bar in bars:
    #     ax.bar_label(bar, fmt=fmt, label_type='edge', padding=2, fontsize='x-small')

    ax.legend(ncol=3)

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(out_fname,
                # bbox_extra_artists=(lgd,),
                bbox_inches='tight')


if __name__ == '__main__':
    # read_data()
    main()

