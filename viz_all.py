import os
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

# constants
TITLES = {
    'mm1k' : 'M/M/1/K',
    'mmmmr': 'M/M/m/m+r',
    'lbwb' : 'Leaky Bucket',
    'cio'  : 'Circular I/O',
}
# // STYLES = [
# //     {'color': 'red'   }, # // 'marker': 'o' 
# //     {'color': 'orange'}, # // 'marker': '^' 
# //     {'color': 'green' }, # // 'marker': 's' 
# //     {'color': 'blue'  }, # // 'marker': '*' 
# //     {'color': 'purple'}, # // 'marker': 'x' 
# // ]

def null_fix(data):
    r"""Fix loss log by early stopping

    Args
    ----
    data : object
        Loss log to be fixed.

    Returns
    -------
    data : object
        Fixed loss log.

    """
    # change nothing
    return data.copy()

def early_stop_fix(data):
    r"""Fix loss log by early stopping

    Args
    ----
    data : object
        Loss log to be fixed.

    Returns
    -------
    data : object
        Fixed loss log.

    """
    # get training and test loss
    data_tr = data['loss_lst_tr']
    data_te = data['loss_lst_te']

    # generate new loss with early stopping
    es_tr, es_te = [], []
    for itr_tr, itr_te in zip(data_tr, data_te):
        if len(es_tr) == 0:
            es_tr.append(itr_tr)
            es_te.append(itr_te)
        elif itr_tr < es_tr[-1]:
            es_tr.append(itr_tr)
            es_te.append(itr_te)
        else:
            es_tr.append(es_tr[-1])
            es_te.append(es_te[-1])
    return {'loss_lst_tr': es_tr, 'loss_lst_te': es_te}

# fixing function
_fix = null_fix

def add_glyph(ax, case, part, num, ctype, magic, alpha_str, label, maxlen=None, **kargs):
    r"""Add a Glyph to given axis

    Args
    ----
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis to add glyph.
    case : str
        Studying case.
    part : str
        Part of data to visualize.
    num : int
        Number of training samples used in studying case.
    ctype : str
        Training criterion used in studying case.
    magic : str
        Numerical method used in studying case.
    alpha_str : str
        Alpha used in studying case in string.
    label : str
        Label of adding glyph.
    maxlen : int
        Maximum length of data to visualize.

    Returns
    -------
    vmin : float
        Valid minimum.
    vmax : float
        Valid maximum.

    """
    # construct path
    rootname = "{}_{}_{}_{}_{}".format(case, 'all', ctype, 'all', alpha_str)
    filename = "{}_{}_{}_{}_{}.pt".format(case, num, ctype, magic, alpha_str)
    path = os.path.join(rootname, filename)

    # load data
    data = _fix(torch.load(path)) # // torch.load(path)
    ydata = data[part]
    maxlen = maxlen or len(ydata)
    freq = maxlen // 10

    # visualize the glyph
    xdata = list(range(maxlen))
    ydata = ydata[0:maxlen]
    ax.plot(xdata, ydata, label=label, markevery=freq, alpha=0.85, lw=2, **kargs)
    return min(ydata), ydata[0]

def best(case):
    r"""Find best configuration for given case

    Args
    ----
    case : str
        Studying case.

    """
    # search for best configuration
    num_lst = [200] # // [1, 5, 10, 25, 50, 100, 200]
    ctype = 'cond'
    magic_lst = ['7', '4', 'rr', 'inf', 'rrinf']
    alpha_str_lst = ['1'] # // ['-1', '0', '0.01', '0.1', '1', '10']
    best_point, best_cfg = None, None
    for num in num_lst:
        for magic in magic_lst:
            for alpha_str in alpha_str_lst:
                rootname = "{}_{}_{}_{}_{}".format(case, 'all', ctype, 'all', alpha_str)
                filename = "{}_{}_{}_{}_{}.pt".format(case, num, ctype, magic, alpha_str)
                path = os.path.join(rootname, filename)
                data = _fix(torch.load(path))
                ydata = data['loss_lst_te']
                point = min(ydata)
                if best_point is None or point < best_point:
                    best_point = point
                    best_cfg = (num, magic, alpha_str)
                else:
                    pass

    # export best configuration
    global CASE
    global TITLE
    global BEST_NUM
    global BEST_MAGIC
    global BEST_ALPHA_STR
    CASE, TITLE = case, TITLES[case]
    BEST_NUM, BEST_MAGIC, BEST_ALPHA_STR = best_cfg
    print(best_point, best_cfg)

def magic():
    r"""Numerical Method Comparison on M/M/1/K"""
    title = "Numerical Method Comparison on {}".format(TITLE)
    filename = "{}_magic".format(CASE)
    # fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax = (None, ax)
    # fig.suptitle(title, fontsize=TITLE_FONT)
    # ax[0].set_title('Train', fontsize=LABEL_FONT)
    # ax[0].set_xlabel('#Epochs', fontsize=LABEL_FONT)
    # ax[0].set_ylabel('Observation Loss', fontsize=LABEL_FONT)
    ax[1].set_title('Test', fontsize=LABEL_FONT)
    ax[1].set_xlabel('#Epochs', fontsize=LABEL_FONT)
    ax[1].set_ylabel('MSE Loss of Focusing State\nSteady Distribution', fontsize=LABEL_FONT)
    case = CASE
    num = BEST_NUM
    ctype = 'cond'
    magic = None
    alpha_str = BEST_ALPHA_STR
    enum_lst = ['7', '4', 'rr', 'inf', 'rrinf']
    label_lst = [r'$2^7$', r'$2^4$', 'RR', 'Inf', 'RR-Inf']
    maxlen = None
    vmin_dict, vmax_dict = dict(tr=None, te=None), dict(tr=None, te=None)
    for i, magic in enumerate(enum_lst):
        style = {} # // STYLES[i]
        label = label_lst[i]
        # vmin, vmax = add_glyph(
        #     ax[0], case, 'loss_lst_tr', num, ctype, magic, alpha_str, label=label, maxlen=maxlen, **style)
        # vmin_dict['tr'] = vmin if vmin_dict['tr'] is None else min(vmin_dict['tr'], vmin)
        # vmax_dict['tr'] = vmax if vmax_dict['tr'] is None else max(vmax_dict['tr'], vmax)
        vmin, vmax = add_glyph(
            ax[1], case, 'loss_lst_te', num, ctype, magic, alpha_str, label=label, maxlen=maxlen, **style)
        vmin_dict['te'] = vmin if vmin_dict['te'] is None else min(vmin_dict['te'], vmin)
        vmax_dict['te'] = vmax if vmax_dict['te'] is None else max(vmax_dict['te'], vmax)
    # ax[0].legend(loc='upper right', fontsize=LEGEND_FONT)
    ax[1].legend(loc='upper right', fontsize=LEGEND_FONT)
    # offset = (vmax_dict['tr'] - vmin_dict['tr']) * 0.05
    # ax[0].set_ylim(vmin_dict['tr'] - offset, vmax_dict['tr'] + offset)
    offset = (vmax_dict['te'] - vmin_dict['te']) * 0.05
    ax[1].set_ylim(vmin_dict['te'] - offset, vmax_dict['te'] + offset)
    fig.savefig(filename)
    plt.close(fig)

def num():
    r"""Number of Training Samples Comparison on M/M/1/K"""
    title = "Number of Training Samples Comparison on {}".format(TITLE)
    filename = "{}_num".format(CASE)
    # fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax = (None, ax)
    # fig.suptitle(title, fontsize=TITLE_FONT)
    # ax[0].set_title('Train', fontsize=LABEL_FONT)
    # ax[0].set_xlabel('#Epochs', fontsize=LABEL_FONT)
    # ax[0].set_ylabel('Observation Loss', fontsize=LABEL_FONT)
    ax[1].set_title('Test', fontsize=LABEL_FONT)
    ax[1].set_xlabel('#Epochs', fontsize=LABEL_FONT)
    ax[1].set_ylabel('MSE Loss of Focusing State\nSteady Distribution', fontsize=LABEL_FONT)
    case = CASE
    num = None
    ctype = 'cond'
    magic = BEST_MAGIC
    alpha_str = BEST_ALPHA_STR
    enum_lst = [1, 5, 10, 25, 50, 100, 200]
    label_lst = enum_lst
    maxlen = None
    vmin_dict, vmax_dict = dict(tr=None, te=None), dict(tr=None, te=None)
    for i, num in enumerate(enum_lst):
        style = {} # // STYLES[i]
        label = label_lst[i]
        # vmin, vmax = add_glyph(
        #     ax[0], case, 'loss_lst_tr', num, ctype, magic, alpha_str, label=label, maxlen=maxlen, **style)
        # vmin_dict['tr'] = vmin if vmin_dict['tr'] is None else min(vmin_dict['tr'], vmin)
        # vmax_dict['tr'] = vmax if vmax_dict['tr'] is None else max(vmax_dict['tr'], vmax)
        vmin, vmax = add_glyph(
            ax[1], case, 'loss_lst_te', num, ctype, magic, alpha_str, label=label, maxlen=maxlen, **style)
        vmin_dict['te'] = vmin if vmin_dict['te'] is None else min(vmin_dict['te'], vmin)
        vmax_dict['te'] = vmax if vmax_dict['te'] is None else max(vmax_dict['te'], vmax)
    # ax[0].legend(loc='upper right', fontsize=LEGEND_FONT)
    ax[1].legend(loc='upper right', fontsize=LEGEND_FONT)
    # offset = (vmax_dict['tr'] - vmin_dict['tr']) * 0.05
    # ax[0].set_ylim(vmin_dict['tr'] - offset, vmax_dict['tr'] + offset)
    offset = (vmax_dict['te'] - vmin_dict['te']) * 0.05
    ax[1].set_ylim(vmin_dict['te'] - offset, vmax_dict['te'] + offset)
    fig.savefig(filename)
    plt.close(fig)

def alpha():
    r"""Prior Strength comparison on M/M/1/K"""
    title = "Prior Strength Comparison on {}".format(TITLE)
    filename = "{}_alpha".format(CASE)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    # fig.suptitle(title, fontsize=TITLE_FONT)
    ax[0].set_title('Train', fontsize=LABEL_FONT)
    ax[0].set_xlabel('#Epochs', fontsize=LABEL_FONT)
    ax[0].set_ylabel('Observation Loss', fontsize=LABEL_FONT)
    ax[1].set_title('Test', fontsize=LABEL_FONT)
    ax[1].set_xlabel('#Epochs', fontsize=LABEL_FONT)
    ax[1].set_ylabel('MSE Loss of Focusing State\nSteady Distribution', fontsize=LABEL_FONT)
    case = CASE
    num = BEST_NUM
    ctype = 'cond'
    magic = BEST_MAGIC
    alpha_str = None
    # enum_lst = ['-1', '0', '0.01', '0.1', '1', '10']
    # label_lst = ['Null'] + enum_lst[1:]
    enum_lst = ['0.01', '0.1', '1', '10', '100']
    label_lst = enum_lst
    maxlen = None
    vmin_dict, vmax_dict = dict(tr=None, te=None), dict(tr=None, te=None)
    for i, alpha_str in enumerate(enum_lst):
        style = {} # // STYLES[i]
        label = label_lst[i]
        vmin, vmax = add_glyph(
            ax[0], case, 'loss_lst_tr', num, ctype, magic, alpha_str, label=label, maxlen=maxlen, **style)
        vmin_dict['tr'] = vmin if vmin_dict['tr'] is None else min(vmin_dict['tr'], vmin)
        vmax_dict['tr'] = vmax if vmax_dict['tr'] is None else max(vmax_dict['tr'], vmax)
        vmin, vmax = add_glyph(
            ax[1], case, 'loss_lst_te', num, ctype, magic, alpha_str, label=label, maxlen=maxlen, **style)
        vmin_dict['te'] = vmin if vmin_dict['te'] is None else min(vmin_dict['te'], vmin)
        vmax_dict['te'] = vmax if vmax_dict['te'] is None else max(vmax_dict['te'], vmax)
    ax[0].legend(loc='upper right', fontsize=LEGEND_FONT)
    ax[1].legend(loc='upper right', fontsize=LEGEND_FONT)
    offset = (vmax_dict['tr'] - vmin_dict['tr']) * 0.05
    ax[0].set_ylim(vmin_dict['tr'] - offset, vmax_dict['tr'] + offset)
    offset = (vmax_dict['te'] - vmin_dict['te']) * 0.05
    ax[1].set_ylim(vmin_dict['te'] - offset, vmax_dict['te'] + offset)
    fig.savefig(filename)
    plt.close(fig)

# run all visulization
# TITLE_FONT = 20
LABEL_FONT = 24
LEGEND_FONT = 20
mpl.rcParams['xtick.labelsize'] = 12
best('mm1k')
magic()
num()
alpha()
best('mmmmr')
magic()
num()
alpha()
best('lbwb')
magic()
num()
alpha()
best('cio')
magic()
num()
alpha()