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
        sg_min, sg_max = 1, 0
        for const_lambd in lambd_cands:
            # generate raw matrix
            X = self.mx(self.noise, const_lambd, *self.mxargs)

            # remember spectral gap
            sg = self.spectral_gap(X)
            sg_min = min(sg_min, sg)
            sg_max = max(sg_max, sg)

            # get steady state
            pi = F.stdy_dist_rrx(X, None, trick='rrinf')
            target = F.stdy_dist_rrx(X, self.ind, trick='rrinf')

            # sample observations from steady state on ideal birth-death process
            probas = target.data.numpy()
            obvs = [stats.poisson.rvs(proba * const_lambd) for proba in probas]
            obvs.append(stats.poisson.rvs((1 - probas.sum()) * const_lambd))
            self.samples.append((torch.Tensor([const_lambd]), pi, torch.Tensor(obvs)))

        # output some data information
        fmt = "{:3d}, {:.3f} ({:.3f}), {:.3f} ({:.3f})"
        sg_min_pow = sg_min ** (2 ** 7)
        sg_max_pow = sg_max ** (2 ** 7)
        print(fmt.format(len(self), sg_min, sg_min_pow, sg_max, sg_max_pow))

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

    def spectral_gap(self, mx):
        r"""Get spectral gap of transition matrix for given matrix

        Args
        ----
        mx : torch.Tensor
            Given matrix.

        Returns
        -------
        sg : float
            Spectral gap.

        """
        # get diagonal line for uniform normalization
        diags = torch.sum(mx, dim=1)
        diags_mx = torch.diagflat(diags)

        # get gamma for uniform normalization
        gamma, _id = torch.max(diags), torch.argmax(diags)
        gamma, _id = gamma.item() + 0.001, _id.item()

        # get P
        Q = mx - diags_mx
        P = torch.eye(self.k, dtype=Q.dtype, device=Q.device) + Q / gamma
        P = P.data.numpy()

        # get second largest eigenvalue
        return np.real(np.sort(np.linalg.eigvals(P)))[-2]


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

    def update_input_prior(self, mx, lambd):
        r"""Construct data matrix

        Args
        ----
        mx : torch.Tensor
            Matrix tensor to update.
        lambd : float
            Input rate lambda.

        Returns
        -------
        mx : torch.Tensor
            Data matrix updated by input.

        """
        # diagonal line should always be zero
        for i in range(min(mx.size())):
            mx.data[i, i] = 0

        # input (upper diagonal) line should be given value
        for i, j in torch.nonzero(self.bmx):
            mx.data[i, j] = lambd
        return mx

    def param_dict(self):
        r"""Return truth parameter corresponding to the model

        Returns
        -------
        param_dict : dict
            Named parameter.

        """
        # name necessary parameter
        return {'mu': self.mxargs[0]}


class DataLBWB(QueueData):
    r"""Leaky Bucket Web Browsing Queue Data"""
    def __init__(self, c, r, b, const_mu, const_bucket, ind, focus, *args, **kargs):
        r"""Initialize the class

        Args
        ----
        c : int
            Number of web users.
        r : int
            Number of request buffer size.
        b : int
            Number of response bucket size.
        const_mu : int
            Constant mu.
        const_bucket : int
            Constant lambda bucket.
        ind : [int, ...]
            Focusing state indices.
        focus : int
            Ultimate focusing state index for test.

        """
        # generate transition list
        self.trans_lst = []
        for i in range(r + 1):
            for j in range(b + 1):
                # pair state
                src_st = (i, j)

                # request increment (exclude waiting)
                if i + 1 <= r:
                    dst_st = (i + 1, j)
                    mnt = max(0, c - i)
                    self.trans_lst.append((src_st, dst_st, mnt, 'r'))
                else:
                    pass

                # bucket increment
                if j + 1 <= b:
                    dst_st = (i, j + 1)
                    mnt = 1
                    self.trans_lst.append((src_st, dst_st, mnt, 'b'))
                else:
                    pass

                # consume decrement
                if i - 1 >= 0 and j - 1 >= 0:
                    dst_st = (i - 1, j - 1)
                    mnt = 1
                    self.trans_lst.append((src_st, dst_st, mnt, 'c'))
                else:
                    pass

        # enumerate paired states
        self.st2idx = {}
        for src_st, dst_st, mnt, _ in self.trans_lst:
            for st in (src_st, dst_st):
                if st in self.st2idx:
                    pass
                else:
                    self.st2idx[st] = len(self.st2idx)

        # state allocation
        self.k = len(self.st2idx)
        self.mxargs = (const_mu, const_bucket)

        # enumerate paired state arguments
        ind = [self.st2idx[itr] for itr in ind]
        focus = self.st2idx[focus]

        # super call
        QueueData.__init__(self, ind=ind, focus=focus, *args, **kargs)

    def gen_share(self):
        r"""Generate sharing matrix"""
        # allocate matrix
        self.rmx = torch.Tensor(self.k, self.k)
        self.bmx = torch.Tensor(self.k, self.k)
        self.cmx = torch.Tensor(self.k, self.k)

        # disable propagation
        self.rmx.requires_grad = False
        self.bmx.requires_grad = False
        self.cmx.requires_grad = False

        # zero all matrix
        self.rmx.zero_()
        self.bmx.zero_()
        self.cmx.zero_()

        # assign sharing values
        for src_st, dst_st, mnt, hd in self.trans_lst:
            i, j = self.st2idx[src_st], self.st2idx[dst_st]
            mx = getattr(self, "{}mx".format(hd))
            mx[i, j] = mnt

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
        for i, j in torch.nonzero(self.rmx):
            mx.data[i, j] = 0
        for i, j in torch.nonzero(self.bmx):
            mx.data[i, j] = 0
        for i, j in torch.nonzero(self.cmx):
            mx.data[i, j] = 0
        return mx

    def mx(self, noise, lambd, mu, bucket):
        r"""Construct data matrix

        Args
        ----
        noise : torch.Tensor
            Noise tensor.
        lambd : float
            Input rate lambda.
        mu : float
            Output rate lambda.
        bucket : float
            Bucket rate lambda.

        Returns
        -------
        X : torch.Tensor
            Data matrix for given lambda.

        """
        # construct matrix
        X = noise + lambd * self.rmx + mu * self.cmx + bucket * self.bmx
        return X

    def update_input_prior(self, mx, lambd):
        r"""Construct data matrix

        Args
        ----
        mx : torch.Tensor
            Matrix tensor to update.
        lambd : float
            Input rate lambda.

        Returns
        -------
        mx : torch.Tensor
            Data matrix updated by input.

        """
        # diagonal line should always be zero
        for i in range(min(mx.size())):
            mx.data[i, i] = 0

        # input (upper diagonal) line should be given value
        for i, j in torch.nonzero(self.rmx):
            mx.data[i, j] = lambd
        return mx

    def param_dict(self):
        r"""Return truth parameter corresponding to the model

        Returns
        -------
        param_dict : dict
            Named parameter.

        """
        # name necessary parameter
        return {'mu': self.mxargs[0], 'bucket': self.mxargs[1]}


class DataCIO(QueueData):
    r"""Circular Input/Output Queue Data"""
    def __init__(self, s, a1, a2, b, const_mu, proba1, ind, focus, *args, **kargs):
        r"""Initialize the class

        Args
        ----
        s : int
            Total number of packets in closure.
        a1 : int
            Number of input buffer 1.
        a2 : int
            Number of input buffer 2.
        b : int
            Number of output buffer.
        const_mu : int
            Constant mu.
        proba1 : float
            Probability of a packet from output buffer to input buffer 1.
        ind : [int, ...]
            Focusing state indices.
        focus : int
            Ultimate focusing state index for test.

        """
        # generate transition list
        self.trans_lst = []
        for i1 in range(a1 + 1):
            for i2 in range(a2 + 1):
                for j in range(b + 1):
                    # pair state
                    if i1 + i2 + j == s:
                        src_st = (i1, i2, j)
                    else:
                        continue
    
                    # input 1 decrement
                    if i1 - 1 >= 0 and j + 1 <= b:
                        dst_st = (i1 - 1, i2, j + 1)
                        mnt = 1
                        self.trans_lst.append((src_st, dst_st, mnt, 'i1o'))
                    else:
                        pass
    
                    # input 2 decrement
                    if i2 - 1 >= 0 and j + 1 <= b:
                        dst_st = (i1, i2 - 1, j + 1)
                        mnt = 1
                        self.trans_lst.append((src_st, dst_st, mnt, 'i2o'))
                    else:
                        pass
    
                    # input 1 increment
                    if i1 + 1 <= a1 and j - 1 >= 0:
                        dst_st = (i1 + 1, i2, j - 1)
                        mnt = 1
                        self.trans_lst.append((src_st, dst_st, mnt, 'oi1'))
                    else:
                        pass
    
                    # input 2 increment
                    if i2 + 1 <= a1 and j - 1 >= 0:
                        dst_st = (i1, i2 + 1, j - 1)
                        mnt = 1
                        self.trans_lst.append((src_st, dst_st, mnt, 'oi2'))
                    else:
                        pass


        # enumerate paired states
        self.st2idx = {}
        for src_st, dst_st, mnt, _ in self.trans_lst:
            for st in (src_st, dst_st):
                if st in self.st2idx:
                    pass
                else:
                    self.st2idx[st] = len(self.st2idx)

        # state allocation
        self.k = len(self.st2idx)
        self.proba1 = proba1
        self.mxargs = (const_mu,)

        # enumerate paired state arguments
        ind = [self.st2idx[itr] for itr in ind]
        focus = self.st2idx[focus]

        # super call
        QueueData.__init__(self, ind=ind, focus=focus, *args, **kargs)

    def gen_share(self):
        r"""Generate sharing matrix"""
        # allocate matrix
        self.i1omx = torch.Tensor(self.k, self.k)
        self.i2omx = torch.Tensor(self.k, self.k)
        self.oi1mx = torch.Tensor(self.k, self.k)
        self.oi2mx = torch.Tensor(self.k, self.k)

        # disable propagation
        self.i1omx.requires_grad = False
        self.i2omx.requires_grad = False
        self.oi1mx.requires_grad = False
        self.oi2mx.requires_grad = False

        # zero all matrix
        self.i1omx.zero_()
        self.i2omx.zero_()
        self.oi1mx.zero_()
        self.oi2mx.zero_()

        # assign sharing values
        for src_st, dst_st, mnt, hd in self.trans_lst:
            i, j = self.st2idx[src_st], self.st2idx[dst_st]
            mx = getattr(self, "{}mx".format(hd))
            mx[i, j] = mnt

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
        for i, j in torch.nonzero(self.i1omx):
            mx.data[i, j] = 0
        for i, j in torch.nonzero(self.i2omx):
            mx.data[i, j] = 0
        for i, j in torch.nonzero(self.oi1mx):
            mx.data[i, j] = 0
        for i, j in torch.nonzero(self.oi2mx):
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
        I = lambd * self.i1omx + lambd * self.i2omx
        O = mu * self.oi1mx * self.proba1 + mu * self.oi2mx * (1 - self.proba1)
        X = noise + I + O
        return X

    def update_input_prior(self, mx, lambd):
        r"""Construct data matrix

        Args
        ----
        mx : torch.Tensor
            Matrix tensor to update.
        lambd : float
            Input rate lambda.

        Returns
        -------
        mx : torch.Tensor
            Data matrix updated by input.

        """
        # diagonal line should always be zero
        for i in range(min(mx.size())):
            mx.data[i, i] = 0

        # input (upper diagonal) line should be given value
        for i, j in torch.nonzero(self.i1omx):
            mx.data[i, j] = lambd
        for i, j in torch.nonzero(self.i2omx):
            mx.data[i, j] = lambd
        return mx

    def param_dict(self):
        r"""Return truth parameter corresponding to the model

        Returns
        -------
        param_dict : dict
            Named parameter.

        """
        # name necessary parameter
        return {'mu': self.mxargs[0]}


r"""
Function
========
- **mm1k**  : Generate M/M/1/K test case
- **mmmmr** : Generate M/M/m/m+r test case
- **lbwb**  : Generate Leaky Bucket Web Browsing test case
- **cio**   : Generate Circular Input/Output test case
"""


def mm1k(rng, num):
    r"""Generate M/M/1/K test case

    Args
    ----
    rng : str
        Range specifier.
    num : int
        Number of training samples.

    Returns
    -------
    seed : int
        Random seed to use.
    train_data : object
        Training data.
    test_data : object
        Test data.

    """
    # global settings decided by data
    np.set_printoptions(precision=8, suppress=True)

    # set random seed
    seed = 47
    if rng == 's':
        mn, mx = 20, 30
    elif rng == 'l':
        mn, mx = 1, 50
    else:
        pass

    # set sharing configuration
    data_kargs = dict(k=20, m=1, const_mu=25, epsilon=1e-4, ind=[0, 1], focus=-1)

    # generate data
    train_data = DataMMmK(n=num, lamin=mn, lamax=mx, seed=seed - 1, **data_kargs)
    test_data  = DataMMmK(n=50 , lamin=1 , lamax=50, seed=seed + 1, **data_kargs)
    return seed, train_data, test_data


def mmmmr(rng, num):
    r"""Generate M/M/m/m+r test case

    Args
    ----
    rng : str
        Range specifier.
    num : int
        Number of training samples.

    Returns
    -------
    seed : int
        Random seed to use.
    train_data : object
        Training data.
    test_data : object
        Test data.

    """
    # global settings decided by data
    np.set_printoptions(precision=8, suppress=True)

    # set random seed
    seed = 47
    if rng == 's':
        mn, mx = 20, 30
    elif rng == 'l':
        mn, mx = 1, 50
    else:
        pass

    # set sharing configuration
    data_kargs = dict(k=20, m=1, const_mu=25, epsilon=1e-4, ind=[0, 1], focus=-1)

    # generate data
    train_data = DataMMmK(n=num, lamin=mn, lamax=mx, seed=seed - 1, **data_kargs)
    test_data  = DataMMmK(n=50 , lamin=1 , lamax=50, seed=seed + 1, **data_kargs)
    return seed, train_data, test_data


def lbwb(rng, num):
    r"""Generate Leaky Bucket Web Browsing test case

    Args
    ----
    rng : str
        Range specifier.
    num : int
        Number of training samples.

    Returns
    -------
    seed : int
        Random seed to use.
    train_data : object
        Training data.
    test_data : object
        Test data.

    """
    # global settings decided by data
    np.set_printoptions(precision=8, suppress=True)

    # set random seed
    seed = 47
    if rng == 's':
        mn, mx = 20, 30
    elif rng == 'l':
        mn, mx = 1, 50
    else:
        pass

    # set sharing configuration
    data_kargs = dict(
        c=5, r=3, b=2, const_mu=25, const_bucket=15, epsilon=1e-4,
        ind=[(0, 0), (0, 1), (1, 0), (1, 1)], focus=(3, 2))

    # generate data
    train_data = DataLBWB(n=num, lamin=mn, lamax=mx, seed=seed - 1, **data_kargs)
    test_data  = DataLBWB(n=50 , lamin=1 , lamax=50, seed=seed + 1, **data_kargs)
    return seed, train_data, test_data


def cio(rng, num):
    r"""Generate Circular Input/Output test case

    Args
    ----
    rng : str
        Range specifier.
    num : int
        Number of training samples.

    Returns
    -------
    seed : int
        Random seed to use.
    train_data : object
        Training data.
    test_data : object
        Test data.

    """
    # global settings decided by data
    np.set_printoptions(precision=8, suppress=True)

    # set random seed
    seed = 47
    if rng == 's':
        mn, mx = 20, 30
    elif rng == 'l':
        mn, mx = 1, 50
    else:
        pass

    # set sharing configuration
    s, a1, a2, b = 6, 4, 4, 2
    ind = [(3, 3, 0), (2, 3, 1), (3, 2, 1), (2, 4, 0), (4, 2, 0)]
    data_kargs = dict(
        s=s, a1=a1, a2=a2, b=b, const_mu=25, proba1=0.5, epsilon=1e-4,
        ind=ind, focus=(2, 2, 2))

    # generate data
    train_data = DataCIO(n=num, lamin=mn, lamax=mx, seed=seed - 1, **data_kargs)
    test_data  = DataCIO(n=50 , lamin=1 , lamax=50, seed=seed + 1, **data_kargs)
    return seed, train_data, test_data
