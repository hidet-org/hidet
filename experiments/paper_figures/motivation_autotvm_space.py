import matplotlib.pyplot as plt
import numpy as np
import os
from common import exec_color, exec_edge_color

script_dir = os.path.dirname(__file__)
dirname = os.path.basename(script_dir)
# out_fname = os.path.join(script_dir, '..', '..', f'{dirname}.pdf')
exp_name = 'motivation_autotvm_schedule_space'
out_fname = os.path.join(script_dir, 'pdfs', f'{exp_name}.pdf')
print(out_fname)

# plt.style.use('ggplot')

font = {'family': 'serif', 'serif': ['Gentium Basic'], 'size': 15}
plt.rc('font', **font)

# plt.rcParams['text.color'] = 'blue'
# plt.rc('text', **{'color': 'black'})


data = [
    79027200, 22579200, 90316800, 352800, 44352000,
    29030400, 10368000, 36864000, 576000, 19008000,
    16896000, 11520000, 2534400, 9123840, 89100,
    4392960, 3953664, 2787840, 232320, 844800,
    462000, 384384, 349440, 253440
]

colors = [
    '#6DB1FF',
    '#00C2A8',
    '#54C45E',
    '#FFE342',
    '#FF8F8F',
    '#FC9432',
]


def main():
    x = np.array(range(1, len(data) + 1))
    y = np.array(data)

    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.bar(x, y, color=exec_color['autotvm'], edgecolor=exec_edge_color['autotvm'], width=0.70)
    ax.set(
        xlabel='Conv2d in ResNet50',
        xlim=(0, len(data) + 1),
        # xticks=[x[i * 5] for i in range(len(x)) if i * 5 < len(x)],
        xticks=[],

        ylabel='Number of schedules',
        ylim=(1e4, 1e8),
        yscale='log',
        yticks=[1e4, 1e5, 1e6, 1e7, 1e8]
    )

    ax.set_xlabel('Conv2d in ResNet50')
    ax.minorticks_off()

    fig.tight_layout()
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    fig.savefig(out_fname)


if __name__ == '__main__':
    main()
