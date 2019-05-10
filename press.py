import numpy as np
import scipy.stats as stats
import torch
import sys
import os
import copy
import itertools
import lib


r"""
Experiment
==========
- **DataMMmKGap**   : Dataset for M/M/m/K Queue With Small Spectral Gap
"""


class DataMMmKGap(lib.WithRandom):
    r"""Dataset for M/M/m/K Queue With Small Spectral Gap"""
    def __init__(self, n, k, m, const_mu, epsilon=1e-4, ind=[-3, -2], focus=-1, *args, **kargs):
        r"""Initialize the class

        Args
        ----
        n : int
            Number of samples.
        k : int
            Number of states. (Number of customers + 1.)
        m : int
            Number of servers.
        const_mu : int
            Constant mu.
        epsilon : int
            Epsilon for noise transition.
        ind : [int, ...]
            Focusing state indices.
        focus : int
            Ultimate focusing state index for test.

        """
        # super call
        lib.WithRandom.__init__(self, *args, **kargs)

        # to get smallest spectral gap
        assert m == 1

        # save necessary attributes
        self.k = k
        self.m = 1
        self._mu = torch.Tensor([const_mu])
        self._epsilon = epsilon
        self.ind = ind
        self.focus = focus

        # create death vector
        self._mu_vec = lib.bd_shrvec(self._mu, self.k - 1, self.m)

        # create noise transition
        self._noise = torch.Tensor(self.k, self.k)
        self._noise.normal_()
        self._noise = lib._zero_tridiag(self._noise)
        self._noise = torch.abs(self._noise)
        self._noise = self._noise / torch.norm(self._noise) * self._epsilon

        # generate samples
        self.samples = []
        lambd_cands = []
        while len(lambd_cands) < n:
            lambd_cands.extend(list(range(const_mu - 15, const_mu + 15)))
        np.random.shuffle(lambd_cands)
        lambd_cands = lambd_cands[0:n]
        lambd_cands = sorted(lambd_cands)
        for const_lambd in lambd_cands:
            # generate birth-death variable vector
            _lambd = torch.Tensor([const_lambd])
            _lambd_vec = lib.bd_shrvec(_lambd, self.k - 1)

            # generate ideal birth-death process
            X = lib.bd_mat(_lambd_vec, self._mu_vec, self._noise)
            pi = lib.stdy_dist_sol(X)
            target = lib.stdy_dist_sol(X, self.ind)

            # sample observations from steady state on ideal birth-death process
            probas = target.data.numpy()
            obvs = [stats.poisson.rvs(proba * _lambd) for proba in probas]
            obvs.append(stats.poisson.rvs((1 - probas.sum()) * _lambd))
            self.samples.append((_lambd, pi, torch.Tensor(obvs)))

    def __len__(self):
        r"""Get length of the class

        Returns
        -------
        length : int
            Length of the class.

        """
        # get length
        return len(self.samples)



r"""
Study
=====
- **MMmmr** : M/M/m/m+r Hyper Parameter Study
- **MM1KG** : M/M/1/K With Small Spectral Gap Hyper Parameter Study
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
        r=M - M // 2, m=M // 2, const_mu=25, epsilon=1e-4, ind=[0, 1], focus=-1)
    data = lib.DataMMmmr(num, seed=seed, **kargs)
    test_data = lib.DataMMmmr(TEST_NUM, seed=seed + 1, **kargs)
    layers = lib.MMmKModule(k=data.k, m=data.m, noise=True)
    return 'mmmmr', (data, test_data), layers


def mm1kg(num, seed):
    r"""Generate M/M/1/K With Small Spectral Gap Data and Model

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
        k=M, m=1, const_mu=25, epsilon=1e-4, ind=[0, 1], focus=-1)
    data = DataMMmKGap(num, seed=seed, **kargs)
    test_data = lib.DataMMmK(TEST_NUM, seed=seed + 1, **kargs)
    layers = lib.MMmKModule(k=data.k, m=data.m, noise=True)
    return 'mm1kg', (data, test_data), layers


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
    comb_cands.append(['pow', 'hav']) # 'sym', 'raw', 
    comb_cands.append(['resi'])
    comb_cands.append(['single'])
    comb_cands.append(['adam'])
    comb_cands.append(['1e-2'])
    comb_cands.append(['1000']) # '0', '1', '30', 
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
    task, num, hyper, M = sys.argv[1:]
    assert task in ('mm1kg', 'mmmmr')
    num = int(num)
    assert hyper in ('quick', 'hyper')
    is_hyper = True if hyper == 'hyper' else False
    assert not is_hyper
    M = int(M)

    # do targeting hyper study
    if sys.argv[1] == 'mm1kg':
        root, data, layers = mm1kg(num, seed=DATA_SEED)
    elif sys.argv[1] == 'mmmmr':
        root, data, layers = mmmmr(num, seed=DATA_SEED)
    else:
        raise RuntimeError()
    root = "{}-{}-{}".format('press', M, root)
    study(root, data, layers, seed=MODEL_SEED)
