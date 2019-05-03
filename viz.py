import torch
import matplotlib.pyplot as plt


r"""
Visualize
=========
- **single_ctype_cfgs** : Visualize results of all configurations for a single ctype
- **each_ctype_best**   : Visualize results of best configurations for each ctype
"""


def single_ctype_cfgs(ctype):
    r"""Visualize results of all configurations for a single ctype

    Args
    ----
    ctype : str
        Criterion type.

    """
    # create canvas
    fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(15, 8))
    ax_tr.set_title('Train Loss')
    ax_tr.set_facecolor('silver')
    ax_tr.set_axisbelow(True)
    ax_tr.grid(axis='y', linestyle='-', linewidth='0.5', color='white')
    ax_te.set_title('Test Loss')
    ax_te.set_facecolor('silver')
    ax_te.set_axisbelow(True)
    ax_te.grid(axis='y', linestyle='-', linewidth='0.5', color='white')

    # set colors
    cmap = plt.get_cmap('gist_rainbow')
    num = 2 * 3 * 3
    colors = [cmap(i / num) for i in range(num)]

    # traverse all combinations
    cnt = 0
    max_tr, max_te = None, None
    min_tr, min_te = None, None
    line_lst, label_lst = [], []
    for btype in ('single', 'full'):
        for otype in ('sgd', 'adam', 'rms'):
            for lr_str in ('1e-1', '1e-2', '1e-3'):
                name = "{}_{}_{}_{}".format(ctype, btype, otype, lr_str)
                lst_tr, lst_te = torch.load("{}_loss_lst.pt".format(name))
                idl_tr, idl_te = torch.load("{}_ideal_loss.pt".format(name))
                int_tr, int_te = lst_tr[0], lst_te[0]
                min_tr = idl_tr if min_tr is None else min(min_tr, idl_tr)
                min_te = idl_te if min_te is None else min(min_te, idl_te)
                max_tr = int_tr if max_tr is None else max(max_tr, int_tr)
                max_te = int_te if max_te is None else max(max_te, int_te)
                xdata_tr, xdata_te = list(range(len(lst_tr))), list(range(len(lst_te)))
                ydata_tr, ydata_te = lst_tr, lst_te
                line = ax_tr.plot(xdata_tr, ydata_tr, color=colors[cnt], label=name)[0]
                line = ax_te.plot(xdata_te, ydata_te, color=colors[cnt], label=name)[0]
                line_lst.append(line)
                label_lst.append(name)
                cnt += 1
    ax_tr.axhline(min_tr, color='white', lw=0.5, ls='--')
    ax_te.axhline(min_te, color='white', lw=0.5, ls='--')

    # reset range
    range_tr, range_te = max_tr - min_tr, max_te - min_te
    ax_tr.set_ylim(min_tr - range_tr * 0.05, max_tr + range_tr * 0.05)
    ax_te.set_ylim(min_te - range_te * 0.05, max_te + range_te * 0.05)

    # legend
    fig.legend(handles=line_lst, labels=label_lst, loc='lower center', borderaxespad=0.1, ncol=3)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig("compare_{}.png".format(ctype))

def each_ctype_best():
    r"""Visualize results of best configurations for each ctype

    Args
    ----
    ctype : str
        Criterion type.

    """
    # create canvas
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_title('Test Loss')
    ax.set_facecolor('silver')
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='-', linewidth='0.5', color='white')

    # set colors
    cmap = plt.get_cmap('gist_rainbow')
    num = 3
    colors = [cmap(i / num) for i in range(num)]

    # traverse all combinations
    vmax, vmin = None, None
    line_lst, label_lst = [], []
    for i, ctype in enumerate(('mse', 'cond', 'resi')):
        best_loss, best_name, best_lst, best_idl = None, None, None, None
        for btype in ('single', 'full'):
            for otype in ('sgd', 'adam', 'rms'):
                for lr_str in ('1e-1', '1e-2', '1e-3'):
                    name = "{}_{}_{}_{}".format(ctype, btype, otype, lr_str)
                    _, lst_te = torch.load("{}_loss_lst.pt".format(name))
                    _, idl_te = torch.load("{}_ideal_loss.pt".format(name))
                    if best_loss is None or min(lst_te) < best_loss:
                        best_loss = min(lst_te)
                        best_name = name
                        best_lst = lst_te
                        best_idl = idl_te
                    else:
                        pass
        vmin = best_idl if vmin is None else min(vmin, best_idl)
        vmax = best_lst[0] if vmax is None else max(vmax, best_lst[0])
        xdata = list(range(len(best_lst)))
        ydata = best_lst
        line = ax.plot(xdata, ydata, color=colors[i], label=best_name)[0]
        line_lst.append(line)
        label_lst.append(best_name)
    ax.axhline(vmin, color='white', lw=0.5, ls='--')

    # reset range
    vrange = vmax - vmin
    ax.set_ylim(vmin - vrange * 0.05, vmax + vrange * 0.05)

    # legend
    fig.legend(handles=line_lst, labels=label_lst, loc='lower center', borderaxespad=0.1, ncol=3)
    fig.subplots_adjust(bottom=0.1)
    fig.savefig('compare.png')

if __name__ == '__main__':
    r"""Main Entrance"""
    # visualize best configuration
    each_ctype_best()

    # visualize each criterion
    single_ctype_cfgs('mse')
    single_ctype_cfgs('cond')
    single_ctype_cfgs('resi')