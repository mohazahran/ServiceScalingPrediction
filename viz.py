import torch
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


r"""
Visualize
=========
"""


def traverse(root, title, options):
    r"""Visualize best test results of each visualizing option combination

    Args
    ----
    root : str
        Root directory to traverse on.
    title : str
        Title for visualization
    options : [object, ...]
        A list of options to traverse on.

    """
    # construct traverse list
    vind, all_lst = [], []
    for i, (opt, flag, _) in enumerate(options):
        if flag:
            vind.append(i)
        else:
            pass
        all_lst.append(opt)
    all_lst = list(itertools.product(*all_lst))

    # traverse on given list
    viz_dict = {}
    best_criterion, best_param = None, None
    for combine in all_lst:
        name = '_'.join(combine)
        viz_key = '+'.join([combine[itr] for itr in vind])
        data = torch.load(os.path.join(root, "{}.pt".format(name)))
        loss_lst_tr   = data['loss_lst_tr']
        loss_lst_te   = data['loss_lst_te']
        ideal_loss_tr = data['ideal_loss_tr']
        ideal_loss_te = data['ideal_loss_te']
        param         = data['param']
        criterion = min(loss_lst_te)
        if viz_key in viz_dict:
            if criterion < viz_dict[viz_key][1]:
                viz_dict[viz_key] = (name, criterion, data)
            else:
                pass
        else:
            viz_dict[viz_key] = (name, criterion, data)
        if best_criterion is None or criterion < best_criterion:
            best_criterion = criterion
            best_param = param
        else:
            pass

    # set colors
    cmap = plt.get_cmap('gist_rainbow')
    num = len(viz_dict)
    colors = [cmap(i / num) for i in range(num)]

    # check availability for training loss
    viz_train = (len(options[2][0]) == 1)

    # create canvas
    ncol = int(np.ceil(float(num) / 18))
    if viz_train:
        fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(ncol * 3 * (1 + 3 * 2), 6))
    else:
        fig, ax_te = plt.subplots(1, 1, figsize=(ncol * 3 * (1 + 3 * 1), 6))
    fig.suptitle(title)
    if viz_train:
        ax_tr.set_xlabel('#Epochs')
        ax_tr.set_facecolor('silver')
        ax_tr.set_axisbelow(True)
        ax_tr.grid(axis='y', linestyle='-', linewidth='0.5', color='white')
    else:
        pass
    ax_te.set_xlabel('#Epochs')
    ax_te.set_facecolor('silver')
    ax_te.set_axisbelow(True)
    ax_te.grid(axis='y', linestyle='-', linewidth='0.5', color='white')

    # traverse visualization keywords
    vmax_tr, vmax_te = None, None
    vmin_tr, vmin_te = None, None
    for i, (key, (name, _, data)) in enumerate(viz_dict.items()):
        loss_lst_tr   = data['loss_lst_tr']
        loss_lst_te   = data['loss_lst_te']
        ideal_loss_tr = data['ideal_loss_tr']
        ideal_loss_te = data['ideal_loss_te']
        vmax_tr = loss_lst_tr[0] if vmax_tr is None else max(loss_lst_tr[0], vmax_tr)
        vmax_te = loss_lst_te[0] if vmax_te is None else max(loss_lst_te[0], vmax_te)
        vmin_tr = ideal_loss_tr if vmin_tr is None else min(ideal_loss_tr, vmin_tr)
        vmin_te = ideal_loss_te if vmin_te is None else min(ideal_loss_te, vmin_te)
        xdata_tr = list(range(len(loss_lst_tr)))
        xdata_te = list(range(len(loss_lst_te)))
        ydata_tr = loss_lst_tr
        ydata_te = loss_lst_te
        if viz_train:
            line = ax_tr.plot(xdata_tr, ydata_tr, color=colors[i], alpha=0.5, label=name)[0]
        else:
            pass
        line_te = ax_te.plot(xdata_te, ydata_te, color=colors[i], alpha=0.5, label=name)[0]

    # reset range
    vrange_tr = vmax_tr - vmin_tr
    vrange_te = vmax_te - vmin_te
    if viz_train:
        ax_tr.set_ylim(vmin_tr - vrange_tr * 0.05, vmax_tr + vrange_tr * 0.05)
        ax_tr.axhline(vmin_tr, color='black', lw=0.5, ls='--')
    else:
        pass
    ax_te.set_ylim(vmin_te - vrange_te * 0.05, vmax_te + vrange_te * 0.05)
    ax_te.axhline(vmin_te, color='black', lw=0.5, ls='--')

    # legend
    if viz_train:
        box_tr = ax_tr.get_position()
        ax_tr.set_position([box_tr.x0 * 0.75, box_tr.y0, box_tr.width * 0.75, box_tr.height])
    else:
        pass
    box_te = ax_te.get_position()
    ax_te.set_position([box_te.x0 * 0.75, box_te.y0, box_te.width * 0.75, box_te.height])
    ax_te.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=ncol)
    name = "{}_{}.png".format(root, ''.join([options[itr][-1] for itr in vind]))
    fig.savefig(name)

    # show parameter result
    if 'mmmmr' in root:
        ideal_mu = 25
        learn_mu = best_param['mu'].data.item()
        print("[Mu]       Learn: {:.3f}; Ideal: {:.3f}".format(learn_mu, ideal_mu))
    elif 'lbwb' in root:
        ideal_lambd_B = 15
        learn_lambd_B = best_param['lambd_B'].data.item()
        ideal_mu = 25
        learn_mu = best_param['mu'].data.item()
        print("[Lambda_B] Learn: {:.3f}; Ideal: {:.3f}".format(learn_lambd_B, ideal_lambd_B))
        print("[Mu]       Learn: {:.3f}; Ideal: {:.3f}".format(learn_mu, ideal_mu))
    elif 'cio' in root:
        ideal_mu = 25
        learn_mu = best_param['mu'].data.item()
        print("[Mu]       Learn: {:.3f}; Ideal: {:.3f}".format(learn_mu, ideal_mu))
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    r"""Main Entrance"""
    # constant options
    QUICK_OPTIONS = {
        'num': (
            'Hyper Parameter Study on Number of Training Samples',
            [
                (['200', '100', '50', '25', '1'], True, 'num'),
                (['sym'], False, 'rr'),
                (['resi'], False, 'crit'),
                (['single'], False, 'bat'),
                (['adam'], False, 'optim'),
                (['1e-2'], False, 'lr'),
                (['1000'], False, 'alpha'),
            ],
        ),
        'alpha': (
            'Hyper Parameter Study on Number of Regularization',
            [
                (['200'], False, 'num'),
                (['sym'], False, 'rr'),
                (['resi'], False, 'crit'),
                (['single'], False, 'bat'),
                (['adam'], False, 'optim'),
                (['1e-2'], False, 'lr'),
                (['0', '1', '1000'], True, 'alpha'),
            ],
        ),
        'rr': (
            'Hyper Parameter Study on Numerical Method',
            [
                (['200'], False, 'num'),
                (['sym', 'raw', 'pow', 'hav'], True, 'rr'),
                (['resi'], False, 'crit'),
                (['single'], False, 'bat'),
                (['adam'], False, 'optim'),
                (['1e-2'], False, 'lr'),
                (['1000'], False, 'alpha'),
            ],
        ),
        'crit': (
            'Hyper Parameter Study on Training Loss',
            [
                (['200'], False, 'num'),
                (['sym'], False, 'rr'),
                (['resi', 'cond', 'mse'], True, 'crit'),
                (['single'], False, 'bat'),
                (['adam'], False, 'optim'),
                (['1e-2'], False, 'lr'),
                (['1000'], False, 'alpha'),
            ],
        ),
    }


    # traverse and visualize on given options
    root, viz = sys.argv[1:]
    options = QUICK_OPTIONS[viz]
    traverse(root, *options)