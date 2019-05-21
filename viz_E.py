import sys
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

path = sys.argv[1]
assert '-1' in path
mx = torch.load(path)['param']['E'].data.numpy()
for i in range(mx.shape[0]):
    for j in range(i, mx.shape[0]):
        mx[i, j] = 0
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
im = ax.imshow(mx)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Rate', rotation=-90, va='bottom')
colors = ['black' for _ in range(mx.shape[0])]
colors[1] = 'blue'
colors[2] = 'blue'
colors[-2] = 'green'
colors[-1] = 'red'
ax.set_xticks(list(range(mx.shape[1])))
ax.set_yticks(list(range(mx.shape[0])))
ax.xaxis.tick_top()
for xlabel, ylabel, color in zip(ax.get_xticklabels(), ax.get_yticklabels(), colors):
    xlabel.set_color(color)
    ylabel.set_color(color)
root = 'rsfigs'
filename = os.path.basename(sys.argv[1]).split('.')[0]
filename = filename + '_mx'
fig.savefig(os.path.join(root, 'pdf', "{}.pdf".format(filename)), format='pdf')
fig.savefig(os.path.join(root, 'png', "{}.png".format(filename)), format='png')
plt.close(fig)