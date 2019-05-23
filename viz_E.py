import sys
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import data

seed, train_data, test_data = getattr(data, 'rtt')('s')
path = sys.argv[1]
if '-1' in path:
    mx = torch.load(path)['param']['E']
    for i in range(mx.size(0)):
        for j in range(i, mx.size(1)):
            mx.data[i, j] = 0
    lambd = train_data.samples[len(train_data) // 2][0].item()
    mx = train_data.update_input_prior(mx, lambd)
    print("Lambda: {}".format(lambd))
else:
    raise NotImplementedError
    # // mu = torch.load(path)['param']['mu'].data.item()
    # // mx = torch.load(path)['param']['E']
    # // for i in range(mx.size(0)):
    # //     for j in range(i, mx.size(1)):
    # //         mx.data[i, j] = 0
    # // for i in range(mx.size(0) - 1):
    # //     mx.data[i + 1, i] = mu
    # // lambd = train_data.samples[len(train_data) // 2][0].item()
    # // mx = train_data.update_input_prior(mx, lambd)
    # // print("Lambda: {}".format(lambd))
mx = mx.data.numpy()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fontsize = 18
im = ax.imshow(mx)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Rate', rotation=-90, va='bottom', fontsize=fontsize)
cbar.ax.tick_params(axis='y', labelsize=fontsize)
colors = ['black' for _ in range(mx.shape[0])]
colors[1] = 'orange'
colors[2] = 'orange'
colors[-2] = 'purple'
colors[-1] = 'red'
ax.xaxis.tick_top()
ax.tick_params(axis='both', labelsize=fontsize)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# // ax.set_xticks(list(range(mx.shape[1])), fontsize=18)
# // ax.set_yticks(list(range(mx.shape[0])), fontsize=18)
# // for xlabel, ylabel, color in zip(ax.get_xticklabels(), ax.get_yticklabels(), colors):
# //     xlabel.set_color(color)
# //     ylabel.set_color(color)
root = 'rsfigs'
filename = os.path.basename(sys.argv[1]).split('.pt')[0]
filename = filename + '_mx'
fig.savefig(os.path.join(root, 'pdf', "{}.pdf".format(filename)), format='pdf')
fig.savefig(os.path.join(root, 'png', "{}.png".format(filename)), format='png')
plt.close(fig)