from typing import Dict, List
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from common import exec_color, exec_fullname


script_dir = os.path.dirname(__file__)
exp_name = 'exp_space_efficiency'
out_fname = os.path.join(script_dir, 'pdfs', f'{exp_name}.pdf')

# plt.style.use('ggplot')

font = {'family': 'serif', 'serif': ['Gentium Basic'], 'size': 15}
plt.rc('font', **font)

colors = {
    # '#6DB1FF',
    # '#00C2A8',
    'autotvm':
        '#54C45E',
    'ansor':
    '#FFE342',
        # '#FF8F8F',
    'hidet':
        '#FC9432',
}
deep_colors = {
    'autotvm': exec_color['autotvm'],
    'ansor': exec_color['ansor'],
    'hidet': exec_color['hidet']
}
# [
#     '#107115',
#     '#008573',
#     '#008A0E',
#     '#FCCE14',
#     '#CC4E00',
# ]

legend_name = {
    'autotvm': 'AutoTVM',
    'ansor': 'Ansor',
    'hidet': 'Hidet'
}

schedulers = [
    'autotvm',
    'ansor',
    'hidet'
]


def load_data():
    files = {
        'autotvm': './data/exp_space_efficiency_data/autotvm.records.json',
        'ansor': './data/exp_space_efficiency_data/ansor.records.json',
        'hidet': './data/exp_space_efficiency_data/hidet.records.txt'
    }
    data = {
        'autotvm': [],
        'ansor': [],
        'hidet': [],
    }

    with open(files['autotvm'], 'r') as f:
        for line in f.readlines():
            record = json.loads(line)
            cost = record['result'][0][0] * 1000.0
            data['autotvm'].append(cost)

    with open(files['ansor'], 'r') as f:
        for line in f.readlines():
            record = json.loads(line)
            cost = record['r'][0][0] * 1000.0
            data['ansor'].append(cost)

    with open(files['hidet'], 'r') as f:
        f.readline()    # head
        f.readline()    # separate line -----
        for line in f.readlines():
            items = line.split()
            data['hidet'].append(float(items[-1]) + (0.039 - 0.034))  # another kernel is used to reduce
    # print(sorted(data['autotvm']))
    # print(sorted(data['ansor']))
    # print(sorted(data['hidet']))
    return data


def draw(data: Dict[str, List[float]]):
    fig: plt.Figure = plt.figure(
        figsize=(5.8, 3.2),
        tight_layout=True
    )
    ax: plt.Axes = fig.add_subplot()

    x_start = 0.020
    x_end = 0.4
    num_bins = 20
    for name, costs in data.items():
        num_exceeded = np.count_nonzero(np.array(costs) > x_end)
        if num_exceeded > 0:
            print(name, 'exceeded', num_exceeded)

    bins = np.linspace(x_start, x_end, num=num_bins)
    ax.hist([data['autotvm'], data['ansor'], data['hidet']], bins, label=['AutoTVM', 'Ansor', 'Hidet'], density=True)

    lgd = ax.legend()
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(out_fname,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def draw_density(data: Dict[str, List[float]]):
    fig: plt.Figure = plt.figure(
        figsize=(5.8, 2.8),
        tight_layout=True
    )
    ax: plt.Axes = fig.add_subplot()

    x_start = 0.035
    x_end = 0.8
    num_samples = 100
    # x_end / x_start = base^num_samples
    base = np.exp(np.log(x_end / x_start) / num_samples)
    x = np.array(x_start * base ** np.arange(num_samples))

    for name in data:
        num_exceeded = np.count_nonzero(np.array(data[name]) > x_end)
        if num_exceeded > 0:
            print(name, 'exceeded', num_exceeded, ', filtered')
        data[name] = list(filter(lambda v: v <= x_end, data[name]))

    from scipy.stats import gaussian_kde

    for name in schedulers:
        costs = data[name]
        density = gaussian_kde(costs)
        start = float(max(x_start, min(costs)) - 0.001)
        end = float(min(x_end, max(costs)))
        # x = np.linspace(start, end, num=1000)
        xx = x[(x >= start) & (x <= end)]
        ax.plot(xx, density(xx), label=exec_fullname[name], color=deep_colors[name])
        ax.fill_between(xx, 0, density(xx), color=deep_colors[name], alpha=0.1)

    ax.annotate('180 sch.', xy=(0.04, 5), color=deep_colors['hidet'], weight='bold')
    ax.annotate('800 sch.', xy=(0.10, 8.5), xytext=(0.095, 11), color=deep_colors['ansor'],
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle="arc3,rad=-0.05",
                    color=deep_colors['ansor'],
                    # shrinkA=-5,
                    # shrinkB=5,
                ),
                weight="bold"
                )
    ax.annotate('1000 sch.', xy=(0.35, 1.3), xytext=(0.30, 3.6), color=deep_colors['autotvm'],
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle="arc3,rad=+0.15",
                    color=deep_colors['autotvm'],
                    shrinkA=0,
                    # shrinkB=5,
                ),
                weight='bold'
                )

    # ax.hist([data['autotvm'], data['ansor'], data['hidet']], bins, label=['AutoTVM', 'Ansor', 'Hidet'], density=True)
    ax.set_ylabel('Schedule Latency\nDistribution Density')
    ax.set_xlabel(r'Latency ($\mu s$)')
    ax.set_xlim(left=x_start, right=x_end)
    ax.set_xscale('log', base=base)
    xticks = [0.039, 0.073, 0.093, x_end]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['{:.0f}'.format(v * 1000.0) for v in xticks])

    lgd = ax.legend()
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(out_fname,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')


if __name__ == '__main__':
    # draw(load_data())
    draw_density(load_data())
