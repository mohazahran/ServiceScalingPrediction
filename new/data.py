import numpy as np
import scipy.stats as stats
import torch
import function as F


r"""
Data
====
- **QueueData** : Queue Data Base
- **MMmKData**  : M/M/m/K Queue Data
- **LBWBData**  : Leaky Bucket Web Browsing Queue Data
- **CIOData**   : Circular Input/Output Queue Data
"""


class QueueData(object):
    r"""Queue Data Base"""
    def __init__(self, lamin, lamax, n, epsilon, ind, focus, seed):
        r"""Initialize the class

        Args
        ----
        n : int
            Number of samples.
        epsilon : int
            Epsilon for noise transition.
        ind : [int, ...]
            Focusing state indices.
        focus : int
            Ultimate focusing state index for test.
        lamin : int
            Minimum (inclusive) of sampled lambda.
        lamax : int
            Maximum (inclusive) of sampled lambda.
        seed : int
            Random seed.

        """
        # configure random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # save necessary attributes
        self.epsilon = epsilon
        self.ind = ind
        self.focus = focus

        # generate sharing matrix
        self.gen_share()

        # generate noisy transition matrix
        self.noise = self.noise_mx()

        # allocate data buffer
        self.samples = []

        # generate lambda candidates
        lambd_cands = []
        while len(lambd_cands) < n:
            lambd_cands.extend(list(range(lamin, lamax + 1)))
        np.random.shuffle(lambd_cands)
        lambd_cands = lambd_cands[0:n]
        lambd_cands = sorted(lambd_cands)

        # sample observations for all lambda candidates
        for const_lambd in lambd_cands:
            # generate raw matrix
            X = self.mx(self.noise, const_lambd, *self.mxargs)

            # get steady state
            pi = F.stdy_dist_rrx(X, None, trick='hav')
            target = F.stdy_dist_rrx(X, self.ind, trick='hav')

            # sample observations from steady state on ideal birth-death process
            probas = target.data.numpy()
            obvs = [stats.poisson.rvs(proba * const_lambd) for proba in probas]
            obvs.append(stats.poisson.rvs((1 - probas.sum()) * const_lambd))
            self.samples.append((torch.Tensor([const_lambd]), pi, torch.Tensor(obvs)))

    def __len__(self):
        r"""Get length of the class

        Returns
        -------
        length : int
            Length of the class.

        """
        # get length
        return len(self.samples)

    def noise_mx(self):
        r"""Generate a noise matrix

        Returns
        -------
        noise : torch.Tensor
            Noise matrix.

        """
        # create noise transition matirx
        noise = torch.Tensor(self.k, self.k)
        noise.normal_()
        noise = self.zero_prior(noise)
        noise = torch.abs(noise)
        noise = noise / torch.norm(noise) * self.epsilon
        return noise


class DataMMmK(QueueData):
    r"""M/M/m/K Queue Data"""
    def __init__(self, k, m, const_mu, *args, **kargs):
        r"""Initialize the class

        Args
        ----
        k : int
            Number of states.
        m : int
            Number of servers.
        const_mu : int
            Constant mu.

        """
        # state allocation
        self.k = k
        self.m = m
        self.mxargs = (const_mu,)

        # super call
        QueueData.__init__(self, *args, **kargs)

    def gen_share(self):
        r"""Generate sharing matrix"""
        # allocate matrix
        self.bmx = torch.Tensor(self.k, self.k)
        self.dmx = torch.Tensor(self.k, self.k)

        # disable propagation
        self.bmx.requires_grad = False
        self.dmx.requires_grad = False

        # zero all matrix
        self.bmx.zero_()
        self.dmx.zero_()

        # assign sharing values
        for i in range(self.k - 1):
            self.bmx[i, i + 1] = 1
            self.dmx[i + 1, i] = min(self.m, i + 1)

    def zero_prior(self, mx):
        r"""Zero out given matrix based on sharing prior

        Args
        ----
        mx : torch.Tensor
            Matrix.

        Returns
        -------
        mx : torch.Tensor
            Zero-out matrix.

        """
        # diagonal line should always be zero
        for i in range(min(mx.size())):
            mx.data[i, i] = 0

        # non-zero sharing position should always be zero
        for i, j in torch.nonzero(self.bmx):
            mx.data[i, j] = 0
        for i, j in torch.nonzero(self.dmx):
            mx.data[i, j] = 0
        return mx

    def mx(self, noise, lambd, mu):
        r"""Construct data matrix

        Args
        ----
        noise : torch.Tensor
            Noise tensor.
        lambd : float
            Input rate lambda.
        mu : float
            Output rate lambda.

        Returns
        -------
        X : torch.Tensor
            Data matrix for given lambda.

        """
        # construct matrix
        X = noise + lambd * self.bmx + mu * self.dmx
        return X