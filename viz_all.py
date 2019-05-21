import os
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

# constants
CASE_TITLES = {
    'mm1k'  : 'M/M/1/K',
    'mmmmr' : 'M/M/m/m+r',
    'lbwb'  : 'Leaky Bucket',
    'cio'   : 'Circular I/O',
    'actual': 'Actual Collection',
    'rtt'   : 'RTT Deduced Collection',
}

RNG_TITLES = {
    'l': 'Wide',
    's': 'Narrow',
}

LOSS_TITLES = {
    'cond': 'Conditional',
    'resi': 'Residual',
}

MAGIC_TITLES = {
    'rrinf': 'Russian Roulette + Infinte Split',
    '7'    : 'Baseline-7',
    '4'    : 'Baseline-4',
    'rr'   : 'Russian Roulette',
    'inf'  : 'Infinte Split',
}

STYLES = [
    {'color': 'green' },
    {'color': 'red'   },
    {'color': 'blue'  },
    {'color': 'orange'},
    {'color': 'purple'},
    {'color': 'gray'  },
    {'color': 'brown' },
]

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
_fix = None

def add_glyph(ax, data, case, rng, loss, num, magic, alpha, label, maxlen=None, **kargs):
    r"""Add a Glyph to given axis

    Args
    ----
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis to add glyph.
    data : str
        Part of data to visualize.
    case : str
        Studying case.
    rng : str
        Lambda range used in studying case.
    loss : str
        Training criterion used in studying case.
    num : int
        Number of training samples used in studying case.
    magic : str
        Numerical method used in studying case.
    alpha : str
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
    if case in ('actual', 'rtt'):
        rootname = "{}_{}".format(case, loss)
        filename = "{}_{}_{}_{}_{}.pt".format(case, rng, loss, magic, alpha)
    else:
        rootname = "{}_{}_{}_{}_{}".format(case, 'all', loss, 'all', alpha)
        filename = "{}_{}_{}_{}_{}_{}.pt".format(case, rng, num, loss, magic, alpha)
    path = os.path.join(rootname, filename)

    # load data
    data_dict = _fix(torch.load(path))
    if data in ('tr', 'te'):
        ydata = data_dict["loss_lst_{}".format(data)]
    else:
        raise NotImplementedError
    maxlen = maxlen or len(ydata)
    freq = maxlen // 10

    # visualize the glyph
    xdata = list(range(maxlen))
    ydata = ydata[0:maxlen]
    ax.plot(xdata, ydata, label=label, markevery=freq, alpha=0.85, lw=2, **kargs)
    return min(ydata), ydata[0]

def best(case, rng, loss):
    r"""Find best configuration for given case

    Args
    ----
    case : str
        Studying case.
    rng : str
        Studying lambda range.
    loss : str
        Studying loss function.

    """
    # // # search for best configuration
    # // num_lst = [5, 10, 25, 50, 100, 200]
    # // ctype = loss
    # // magic_lst = ['rrinf', '7', '4', 'rr', 'inf', ]
    # // alpha_str_lst = ['-1', '0', '0.01', '0.1', '1', '10', '100']
    # // best_point, best_cfg = None, None
    # // for num in num_lst:
    # //     for magic in magic_lst:
    # //         for alpha_str in alpha_str_lst:
    # //             rootname = "{}_{}_{}_{}_{}".format(case, 'all', ctype, 'all', alpha_str)
    # //             filename = "{}_{}_{}_{}_{}_{}.pt".format(case, rng, num, ctype, magic, alpha_str)
    # //             path = os.path.join(rootname, filename)
    # //             data = _fix(torch.load(path))
    # //             ydata = data['loss_lst_te']
    # //             point = min(ydata)
    # //             if best_point is None or point < best_point:
    # //                 best_point = point
    # //                 best_cfg = (num, magic, alpha_str)
    # //             else:
    # //                 pass

    # force to use constant configuration
    best_point = float('nan')
    best_cfg = (200, 'rrinf', '100')

    # export best configuration
    print('==[ BEST ]==')
    print("Case  : {}".format(CASE_TITLES[case]))
    print("Range : {}".format(RNG_TITLES[rng]))
    print("Loss  : {}".format(LOSS_TITLES[loss]))
    print("Number: {}".format(best_cfg[0]))
    print("Magic : {}".format(MAGIC_TITLES[best_cfg[1]]))
    print("Alpha : {}".format(best_cfg[2]))
    print("Result: {:.8f}".format(best_point))
    return dict(
        case=case, rng=rng, loss=loss, num=[best_cfg[0]], magic=[best_cfg[1]],
        alpha=[best_cfg[2]])

def viz(data, case, rng, loss, num, magic, alpha, num_lst=None, magic_lst=None, alpha_lst=None,
        maxlen=None):
    r"""Comparison on given configuration"""
    # determine title
    if num_lst is not None:
        prefix = '#Training Observations'
        suffix = 'num'
        label_lst = num_lst.copy()
    elif magic_lst is not None:
        prefix = 'Numerical Method'
        suffix = 'magic'
        translate = {
            'rrinf': 'RR-Inf',
            '7'    : '$2^7$',
            '4'    : '$2^4$',
            'rr'   : 'RR',
            'inf'  : 'Inf',
        }
        label_lst = [translate[itr] for itr in magic_lst]
    elif alpha_lst is not None:
        prefix = 'Prior Regularization Strength'
        suffix = 'alpha'
        label_lst = alpha_lst.copy()
        for i in range(len(label_lst)):
            if label_lst[i] == '-1':
                label_lst[i] = 'No Prior'
            else:
                pass
    else:
        raise RuntimeError()
    title = "{prefix} Comparison on {loss} {case} {rng}".format(
        prefix=prefix, loss=LOSS_TITLES[loss], case=CASE_TITLES[case], rng=RNG_TITLES[rng])
    filename = "{case}_{rng}_{loss}_{suffix}_{data}".format(
        case=case, rng=rng, loss=loss, suffix=suffix, data=data)

    # font
    font = dict(fontsize=18)
    leg_font = dict(fontsize=15)

    # allocate canvas
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlabel('#Epochs', **font)
    if data == 'te':
        ax.set_ylabel('MSE Loss on Test-Focusing State', **font)
    elif data == 'tr':
        ax.set_ylabel("{} Loss on Train-Observed States".format(LOSS_TITLES[loss]), **font)
    elif data == 'mu':
        ax.set_ylabel('Training Service Rate $\hat{\mu}$', **font)
    elif data == 'bucket':
        ax.set_ylabel('Training Bucket Fetching Rate $\hat{\lambda_\text{B}}$', **font)
    else:
        raise RuntimeError()
    ax.tick_params(axis='both', which='major', labelsize=font['fontsize'])
    ax.tick_params(axis='both', which='minor', labelsize=font['fontsize'])

    # configure enumeration
    num_lst   = num_lst or num
    magic_lst = magic_lst or magic
    alpha_lst = alpha_lst or alpha
    enum_lst = itertools.product(num_lst, magic_lst, alpha_lst)
    
    # traverse enumeration
    vinf, vsup = None, None
    for i, (num, magic, alpha) in enumerate(enum_lst):
        style = STYLES[i]
        label = label_lst[i]
        vmin, vmax = add_glyph(
            ax, data, case, rng, loss, num, magic, alpha, label=label, maxlen=maxlen, **style)
        vinf = vmin if vinf is None else min(vmin, vinf)
        vsup = vmax if vsup is None else max(vmax, vsup)
    offset = (vsup - vinf) * 0.05
    ax.set_ylim(vinf - offset, vsup + offset)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, fancybox=True, **leg_font)

    # save visualization
    root = 'rsfigs'
    fig.savefig(os.path.join(root, 'pdf', "{}.pdf".format(filename)), format='pdf')
    fig.savefig(os.path.join(root, 'png', "{}.png".format(filename)), format='png')
    plt.close(fig)

# run all visulization
if __name__ == '__main__':
    _fix = null_fix

    cfgs = [
        best('mm1k' , 's', 'cond'),
        best('mmmmr', 's', 'cond'),
        best('lbwb' , 's', 'cond'),
        best('cio'  , 's', 'cond'),
        best('lbwb' , 's', 'resi'),
        best('cio'  , 's', 'resi'),
        best('mm1k' , 'l', 'cond'),
        best('mmmmr', 'l', 'cond'),
        best('lbwb' , 'l', 'cond'),
        best('cio'  , 'l', 'cond'),
        best('lbwb' , 'l', 'resi'),
        best('cio'  , 'l', 'resi')]

    for cfg in cfgs:
        viz('te', **cfg, num_lst=[5, 10, 25, 50, 100, 200])
        viz('te', **cfg, magic_lst=['rrinf', '7', '4', 'rr', 'inf'])
        viz('te', **cfg, alpha_lst=['0.01', '0.1', '1', '10', '100'])

        viz('tr', **cfg, alpha_lst=['0.01', '0.1', '1', '10', '100'])

    viz('te', 'actual', 'l', 'cond', [None], [None], ['100'], magic_lst=['rrinf', '7', '4', 'rr', 'inf'])
    viz('te', 'actual', 's', 'cond', [None], [None], ['100'], magic_lst=['rrinf', '7', '4', 'rr', 'inf'])
    viz('te', 'rtt', 'l', 'cond', [None], [None], ['100'], magic_lst=['rrinf', '7', '4', 'rr', 'inf'])
    viz('te', 'rtt', 's', 'cond', [None], [None], ['100'], magic_lst=['rrinf', '7', '4', 'rr', 'inf'])

    viz('te', 'rtt', 's', 'cond', [None], ['rrinf'], [None], alpha_lst=['-1', '0.01', '100'])
    viz('tr', 'rtt', 's', 'cond', [None], ['rrinf'], [None], alpha_lst=['-1', '0.01', '100'])