import os
import torch
import matplotlib.pyplot as plt

# constants
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

def add_glyph(ax, case, num, ctype, magic, alpha_str, label, maxlen=None, **kargs):
    r"""Add a Glyph to given axis

    Args
    ----
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis to add glyph.
    case : str
        Studying case.
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

    """
    # construct path
    rootname = "{}_{}_{}_{}_{}".format(case, 'all', ctype, 'all', alpha_str)
    filename = "{}_{}_{}_{}_{}.pt".format(case, num, ctype, magic, alpha_str)
    path = os.path.join(rootname, filename)

    # load data
    data = _fix(torch.load(path)) # // torch.load(path)
    ydata = data['loss_lst_te']
    maxlen = maxlen or len(ydata)
    freq = maxlen // 10

    # visualize the glyph
    xdata = list(range(maxlen))
    ydata = ydata[0:maxlen]
    ax.plot(xdata, ydata, label=label, markevery=freq, alpha=0.5, **kargs)

def best(case):
    r"""Find best configuration for given case

    Args
    ----
    case : str
        Studying case.

    """
    # search for best configuration
    case = 'mm1k'
    num_lst = [1, 5, 10, 25, 50, 100, 200]
    ctype = 'cond'
    magic_lst = ['pow', 'rr', 'inf', 'rrinf']
    alpha_str_lst = ['-1', '0', '0.01', '0.1', '1', '10']
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
    global BEST_NUM
    global BEST_MAGIC
    global BEST_ALPHA_STR
    BEST_NUM, BEST_MAGIC, BEST_ALPHA_STR = best_cfg
    print(best_point, best_cfg)

def mm1k_magic():
    r"""Numerical Method Comparison on M/M/1/K"""
    title = 'Numerical Method Comparison on M/M/1/K'
    filename = 'mm1k_magic'
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(title)
    ax.set_xlabel('#Epochs')
    ax.set_ylabel('MSE Loss of Last State Steady Distribution')
    case = 'mm1k'
    num = BEST_NUM
    ctype = 'cond'
    magic = None
    alpha_str = BEST_ALPHA_STR
    enum_lst = ['pow', 'rr', 'inf', 'rrinf']
    label_lst = ['Power', 'RR', 'Inf', 'RR-Inf']
    maxlen = None
    for i, magic in enumerate(enum_lst):
        style = {} # // STYLES[i]
        label = label_lst[i]
        add_glyph(ax, case, num, ctype, magic, alpha_str, label=label, maxlen=maxlen, **style)
    ax.legend()
    fig.savefig(filename)
    plt.close(fig)

def mm1k_num():
    r"""Number of Training Samples Comparison on M/M/1/K"""
    title = 'Number of Training Samples Comparison on M/M/1/K'
    filename = 'mm1k_num'
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(title)
    ax.set_xlabel('#Samples')
    ax.set_ylabel('MSE Loss of Last State Steady Distribution')
    case = 'mm1k'
    num = None
    ctype = 'cond'
    magic = BEST_MAGIC
    alpha_str = BEST_ALPHA_STR
    enum_lst = [1, 5, 10, 25, 50, 100, 200]
    label_lst = enum_lst
    maxlen = None
    for i, num in enumerate(enum_lst):
        style = {} # // STYLES[i]
        label = label_lst[i]
        add_glyph(ax, case, num, ctype, magic, alpha_str, label=label, maxlen=maxlen, **style)
    ax.legend()
    fig.savefig(filename)
    plt.close(fig)

def mm1k_alpha():
    r"""Prior Strength comparison on M/M/1/K"""
    title = 'Prior Strength Comparison on M/M/1/K'
    filename = 'mm1k_alpha'
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(title)
    ax.set_xlabel('#Samples')
    ax.set_ylabel('MSE Loss of Last State Steady Distribution')
    case = 'mm1k'
    num = BEST_NUM
    ctype = 'cond'
    magic = BEST_MAGIC
    alpha_str = None
    enum_lst = ['-1', '0', '0.01', '0.1', '1', '10']
    label_lst = ['Null'] + enum_lst[1:]
    maxlen = None
    for i, alpha_str in enumerate(enum_lst):
        style = {} # // STYLES[i]
        label = label_lst[i]
        add_glyph(ax, case, num, ctype, magic, alpha_str, label=label, maxlen=maxlen, **style)
    ax.legend(loc='upper right')
    fig.savefig(filename)
    plt.close(fig)

# run all visulization
best('mm1k')
mm1k_magic()
mm1k_num()
mm1k_alpha()