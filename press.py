import numpy as np
import torch
import sys
import os
import copy
import itertools
import lib
# // from .lib import _russian_roulette_sym
# // from .lib import _russian_roulette_raw
# // from .lib import _russian_roulette_hav
# // from .lin import GenericSteadyDist
# // from .lib import MMmKModule, LBWBModule, CIOModule
# // from .lib import DataMMmK, DataMMmmr, DataLBWB, DataCIO
# // from .lib import Task


r"""
Study
=====
- **MMmmr** : M/M/m/m+r Hyper Parameter Study
- **LBWB**  : Leaky Bucket Web Browsing Pressure Study
"""


def mmmmr(num, seed):
    r"""Generate M/M/m/m+r Data and Model

    Args
    ----
    num : int
        Number of samples
    seed : int
        Random seed

    Returns
    -------
    root : str
        Root directory.
    data : object
        Dataset.
    layers : torch.nn.Module
        Neural network layers.

    """
    # generate data
    kargs = dict(
        r=13, m=25, const_mu=25, epsilon=1e-4, ind=[0, 1], focus=-1)
    data = lib.DataMMmmr(num, seed=seed, **kargs)
    test_data = lib.DataMMmmr(TEST_NUM, seed=seed + 1, **kargs)
    layers = lib.MMmKModule(k=data.k, m=data.m, noise=True)
    return 'mmmmr', (data, test_data), layers


def study(root, data, layers, seed=47):
    r"""Hyper Parameter Study

    Args
    ----
    root : str
        Root directory.
    data : object
        Dataset.
    layers : torch.nn.Module
        Neural network layers.
    seed : int
        Random seed.

    """
    # clean results folder
    if os.path.isdir(root):
        pass
    else:
        os.makedirs(root)

    # save initial parameters
    len_data = len(data[0])
    init_params = copy.deepcopy(layers.state_dict())

    # some other settings
    rr_dict = dict(
        sym=lib._russian_roulette_sym, raw=lib._russian_roulette_raw,
        hav=lib._russian_roulette_hav)
    stdy_dict = dict(
        sol=lib.stdy_dist_sol, pow=lib.stdy_dist_pow)

    # traverse loss and batch settings
    comb_cands = []
    comb_cands.append(['sym', 'raw', 'pow', 'hav'])
    comb_cands.append(['resi'])
    comb_cands.append(['single'])
    comb_cands.append(['adam'])
    comb_cands.append(['1e-2'])
    comb_cands.append(['1000'])
    hyper_combs = itertools.product(*comb_cands)
    num_epochs  = NUM_EPOCHS
    for combine in hyper_combs:
        dtype, ctype, btype, otype, lr_str, alpha_str = combine
        name = "{}_{}_{}_{}_{}_{}_{}".format(len_data, *combine)
        print("==[{}]==".format(name))
        if dtype == 'sym':
            lib.GenericSteadyDist.RR = lib._russian_roulette_sym
            lib.stdy_dist = lib.stdy_dist_sol
        elif dtype == 'raw':
            lib.GenericSteadyDist.RR = lib._russian_roulette_raw
            lib.stdy_dist = lib.stdy_dist_sol
        elif dtype == 'pow':
            lib.GenericSteadyDist.RR = None
            lib.stdy_dist = lib.stdy_dist_pow
        elif dtype == 'hav':
            lib.GenericSteadyDist.RR = lib._russian_roulette_hav
            lib.stdy_dist = lib.stdy_dist_sol
        else:
            raise RuntimeError()
        layers.load_state_dict(init_params)
        lr, alpha = float(lr_str), float(alpha_str)
        task = lib.Task(
            data[0], data[1], layers, ctype, otype, btype, lr=lr, alpha=alpha, seed=seed)
        task.fit_from_rand(num_epochs, root=root, name=name)


if __name__ == '__main__':
    r"""Main Entrance"""
    # set precision
    np.set_printoptions(precision=8, suppress=True)
    torch.Tensor = torch.DoubleTensor

    # set fit epochs
    TEST_NUM = 400
    DATA_SEED = 47
    MODEL_SEED = 47
    NUM_EPOCHS = 100

    # parse arguments
    task, num = sys.argv[1:]
    assert task in ('mm1k', 'mmmmr', 'lbwb', 'cio')
    num = int(num)

    # do targeting hyper study
    if sys.argv[1] == 'mm1k':
        root, data, layers = mm1k(num, seed=DATA_SEED)
    elif sys.argv[1] == 'mmmmr':
        root, data, layers = mmmmr(num, seed=DATA_SEED)
    elif sys.argv[1] == 'lbwb':
        root, data, layers = lbwb(num, seed=DATA_SEED)
    elif sys.argv[1] == 'cio':
        root, data, layers = cio(num, seed=DATA_SEED)
    else:
        raise RuntimeError()
    root = "{}-{}".format('press', root)
    study(root, data, layers, seed=MODEL_SEED)