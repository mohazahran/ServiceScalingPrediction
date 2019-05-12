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
    data = torch.load(path)
    ydata = data['loss_lst_te']
    maxlen = maxlen or len(ydata)
    freq = maxlen // 10

    # visualize the glyph
    xdata = list(range(maxlen))
    ydata = ydata[0:maxlen]
    ax.plot(xdata, ydata, label=label, markevery=freq, alpha=0.5, **kargs)

def mm1k_magic():
    r"""Numerical Method Comparison on M/M/1/K"""
    title = 'Numerical Method Comparison on M/M/1/K'
    filename = 'mm1k_magic'
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.set_xlabel('#Epochs')
    ax.set_ylabel('MSE Loss of Last State Steady Distribution')
    case = 'mm1k'
    num = 100
    ctype = 'cond'
    magic = None
    alpha_str = '-1'
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
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.set_xlabel('#Samples')
    ax.set_ylabel('MSE Loss of Last State Steady Distribution')
    case = 'mm1k'
    num = None
    ctype = 'cond'
    magic = 'rrinf'
    alpha_str = '-1'
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

# run all visulization
mm1k_magic()
mm1k_num()