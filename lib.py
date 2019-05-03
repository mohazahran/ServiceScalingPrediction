import numpy as np
import scipy.stats as stats
import torch
import re
import copy
import itertools


r"""
Utils
=====
- **_numpy_solve** : Solve linear system by numpy
- **_torch_solve** : Solve linear system by torch
- **_steady_dist** : Get steady state distributuion
"""


def _numpy_solve(A, b):
    r"""Solve Ax = b by numpy

    Args
    ----
    A : torch.Tensor
        Right matrix.
    b : torch.Tensor
        Left matrix.

    Returns
    -------
    x : torch.Tensor
        Target matrix.

    """
    # convert to numpy
    _A = A.cpu().data.numpy()
    _b = b.cpu().data.numpy()

    # solve x
    _x = np.linalg.lstsq(_A, _b, rcond=None)[0]
    x = torch.from_numpy(_x).to(A.device)
    return x


def _torch_solve(A, b):
    r"""Solve Ax = b by torch

    Args
    ----
    A : torch.Tensor
        Right matrix.
    b : torch.Tensor
        Left matrix.

    Returns
    -------
    x : torch.Tensor
        Target matrix.

    """
    # solve x
    return torch.gesv(b, A)[0]


def _steady_dist(P, solver=_numpy_solve):
    r"""Get steady state distribution

    Args
    ----
    P : torch.Tensor
        Transition matrix.
    solver : func
        Function to solve linear system

    Returns
    -------
    pi : torch.Tensor
        Steady state distribution vector.

    """
    # use the trick to get pi by linear system
    k = P.size(1)
    A = P - torch.eye(k, dtype=P.dtype, device=P.device)
    A = torch.cat([A, torch.ones(k, 1, dtype=A.dtype, device=A.device)], dim=1).t()
    b = torch.ones(k + 1, 1, device=A.device)
    b[-1] = 1
    pi = solver(A, b).t()
    return pi


def _russian_roulette_sym(P, Pinf, M, dist, *args, **kargs):
    r"""Russian Roulette infinite product summation with infinite split trick

    Args
    ----
    P : torch.Tensor
        Stochastic matrix to apply infinite product summation.
    Pinf : torch.Tensor
        Infinite power of input stochastic matrix.
    M : torch.Tensor
        Midterm matrix in each term of infinite product summation.
    dist : scipy.stats.rv_discrete
        Random variable distribution to sample from for Russian Roulette.

    Returns
    -------
    Y : torch.Tensor
        Infinite product summation result.

    """
    # get shape setting
    k = P.size(1)

    # sample x for Russian Roulette
    x = dist.rvs(*args, **kargs)

    # compute expectation with infinity split
    prod = torch.eye(k, dtype=P.dtype, device=P.device)
    E = torch.zeros(k, k, dtype=P.dtype, device=P.device)
    for i in range(1, x + 1):
        cdf_above = stats.geom.sf(i, *args, **kargs)
        E = E + torch.div(prod, cdf_above)
        prod = torch.matmul(P, prod)
    part1 = torch.matmul(torch.matmul(Pinf, M), E)
    part2 = torch.matmul(torch.matmul(E, M), Pinf)
    return part1 + part2


def _zero_tridiag(mx):
    r"""Zero-out tridiagonal lines for a matrix

    Args
    ----
    mx : torch.Tensor
        Matrix to zero-out tridiagonal lines.

    Returns
    -------
    mx : torch.Tensor
        Matrix whose tridiagonal lines being zero-out.

    """
    # get shape setting
    k = mx.size(1)
    mx = mx.clone()

    # zero-out tridiagonal lines
    for i in range(k - 1):
        mx[i, i + 1] = 0
        mx[i + 1, i] = 0
    for i in range(k):
        mx[i, i] = 0
    return mx


r"""
Autograd
========
- **GenericSteadyDist**  : Differentiable steady state distribution function for generic queue process
- **BirthDeathMat**      : Differentiable construction function for birth-death process matrix
- **BirthDeathShareVec** : Differentiable construction function for parameter sharing birth-death vector
"""


class GenericSteadyDist(torch.autograd.Function):
    r"""Differentiable steady state distribution function for generic queue process"""
    # constants
    GEO_P = 0.1

    @staticmethod
    def forward(ctx, X, ind=None, c=0.001, vmin=1e-20, vmax=1, solver=_numpy_solve):
        r"""Forwarding

        Args
        ----
        X : torch.Tensor
            Queue matrix.
        ind : None or int or [int, ...]
            Specify the indices to select.
        c : float
            Uniform normalization offset.
        vmin : float
            Lower bound of output dimensions.
        vmax : float
            Upper bound of output dimensions.
        solver : func
            Linear sysmtem solver.

        Returns
        -------
        pi : torch.Tensor
            Steady state distribution vector.

        """
        # assertion
        assert len(X.size()) == 2
        assert len(torch.nonzero(torch.diagonal(X))) == 0

        # get necessary attributes
        k = X.size(1)
        device = X.device
        ind = [ind] if isinstance(ind, int) else (ind or list(range(k)))

        # get diagonal line for uniform normalization
        diags = torch.sum(X, dim=1)
        diags_mx = torch.diagflat(diags)

        # get gamma for uniform normalization
        gamma, _id = torch.max(diags), torch.argmax(diags)
        gamma, _id = gamma.item() + c, _id.item()

        # get Q
        Q = X - diags_mx
        P = torch.eye(k, dtype=Q.dtype, device=Q.device) + Q / gamma

        # use the trick to get pi by linear system
        pi = _steady_dist(P, solver)
        pi = torch.clamp(pi, vmin, vmax).squeeze()

        # cache necessary attributes for backwarding
        ctx.k     = k
        ctx.gamma = gamma
        ctx._id   = _id
        ctx.Q     = Q
        ctx.P     = P
        ctx.pi    = pi
        ctx.ind   = ind

        # return selected distribution
        return pi[ind]

    @staticmethod
    def backward(ctx, grad_from_prev):
        r"""Backwarding

        Args
        ----
        grad_from_prev : torch.Tensor
            Cumulative gradients from previous steps.

        Returns
        -------
        grad_to_next : torch.Tensor
            Cumulative gradients to next step.

        """
        # fetch necessary attributes to local
        k     = ctx.k
        gamma = ctx.gamma
        _id   = ctx._id
        Q     = ctx.Q
        P     = ctx.P
        pi    = ctx.pi
        ind   = ctx.ind
        cls   = GenericSteadyDist

        # construct selection gradient factor
        fact_S = cls._fact_S(k, ind, dtype=P.dtype, device=P.device)

        # construct uniform normalization gradient coefficient
        coeff_gamma = cls._coeff_gamma(k, _id, dtype=P.dtype, device=P.device)

        # construct P construction gradient coefficient
        coeff_P = cls._coeff_P(k, P, Q, gamma, coeff_gamma, dtype=P.dtype, device=P.device)

        # construct selection gradient coefficient
        coeff_S = cls._coeff_S(k, P, pi, coeff_P)

        # integrate all gradients, coefficients and factors
        grad_to_X = cls._integrate(grad_from_prev, fact_S, coeff_S)
        grad_to_X = grad_to_X.view(k, k)
        return grad_to_X, None, None, None, None, None

    @staticmethod
    def _fact_S(k, ind, dtype, device):
        r"""Construct selection gradient factor

        Args
        ----
        k : int
            Number of transition states.
        ind : [int, ...]
            Selected indices.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            Tensor dtype.

        Returns
        ------
        fact : torch.Tensor
            Selection gradient factor tensor.

        """
        # construct selection gradient factor
        fact_S = torch.zeros(len(ind), k, k, dtype=dtype, device=device)
        for i in range(len(ind)):
            fact_S[i, :, ind[i]] = 1 / k
        return fact_S

    @staticmethod
    def _coeff_gamma(k, idx, dtype, device):
        r"""Construct uniform normalization gradient coefficient

        Args
        ----
        k : int
            Number of transition states.
        idx : int
            Row index used to generate gamma.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            Tensor dtype.

        Returns
        ------
        coeff : torch.Tensor
            Uniform normalization gradient coefficient tensor.

        """
        # construct uniform normalization gradient coefficient
        coeff_gamma = torch.zeros(k, dtype=dtype, device=device)
        coeff_gamma[idx] = 1
        return coeff_gamma

    @staticmethod
    def _coeff_P(k, P, Q, gamma, coeff_gamma, dtype, device):
        r"""Construct uniform normalization gradient coefficient

        Args
        ----
        k : int
            Number of transition states.
        P : torch.Tensor
            Tensor P.
        Q : torch.Tensor
            Tensor Q.
        gamma : float
            Gamma.
        coeff_gamma : torch.Tensor
            Uniform normalization gradient coefficient tensor.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            Tensor dtype.

        Returns
        ------
        coeff : torch.Tensor
            Uniform normalization gradient coefficient tensor.

        """
        # construct Q construction gradient coefficient
        coeff_Q = torch.zeros(k, k, k, k, dtype=dtype, device=device)
        for i in range(k):
            for j in range(k):
                if j == i:
                    coeff_Q[i, j, i, j] = 0
                else:
                    coeff_Q[i, j, i, j] = 1
                    coeff_Q[i, j, i, i] = -1

        # construct P construction gradient coefficient
        coeff_P = torch.zeros(k, k, k, k, dtype=dtype, device=device)
        for i in range(k):
            for j in range(k):
                coeff_P[i, j] = torch.mul(coeff_Q[i, j], gamma) - torch.mul(Q, coeff_gamma[i]) 
                coeff_P[i, j] = torch.div(coeff_P[i, j], gamma ** 2)
        coeff_P = coeff_P.view(k * k, k, k)
        return coeff_P

    @staticmethod
    def _coeff_S(k, P, pi, coeff_P):
        r"""Construct selection gradient coefficient

        Args
        ----
        k : int
            Number of transition states.
        P : torch.Tensor
            Tensor P.
        pi : torch.Tensor
            Steady state distribution vector.
        coeff_P : torch.Tensor
            Uniform normalization gradient coefficient tensor.

        Returns
        ------
        coeff : torch.Tensor
            Selection gradient coefficient tensor.

        """
        # construct selection gradient coefficient
        Pinf = pi.repeat((k, 1))
        geo_p = GenericSteadyDist.GEO_P
        coeff_S = _russian_roulette_sym(P, Pinf, coeff_P, stats.geom, geo_p)
        return coeff_S

    @staticmethod
    def _integrate(grad_from_prev, fact, coeff):
        r"""Integrate all gradients, coefficients and factors

        Args
        ----
        grad_from_prev : torch.Tensor
            Cumulative gradients from previous steps.
        fact : torch.Tensor
            Selection gradient factor tensor.
        coeff : torch.Tensor
            Selection gradient coefficient tensor.

        Returns
        ------
        grad : torch.Tensor
            Integrated gradient tensor,

        """
        # integrate all gradients, coefficients and factors
        grad_lst = []
        for grad_itr, fact_itr in zip(grad_from_prev, fact):
            grad = grad_itr * torch.sum(fact_itr * coeff, (1, 2))
            grad_lst.append(grad)
        grad = torch.stack(grad_lst).sum(dim=0)
        return grad


class BirthDeathMat(torch.autograd.Function):
    r"""Construct birth-death queue matrix supporting noise"""
    @staticmethod
    def forward(ctx, bvec, dvec, E=None):
        r"""Forwarding

        Args
        ----
        bvec : torch.Tensor
            Birth vector.
        dvec : torch.Tensor
            Death vector.
        E : None or torch.Tensor
            Noise queue transaction tensor.

        Returns
        -------
        X : torch.Tensor
            Queue tensor.

        """
        # construct queue matrix
        B = torch.diagflat(bvec, 1)
        D = torch.diagflat(dvec, -1)

        # operate according to noise
        if E is None:
            ctx.noisy = False
            X = B + D
        else:
            ctx.noisy = True
            X = B + D + E
        return X

    @staticmethod
    def backward(ctx, grad_from_prev):
        r"""Backwarding

        Args
        ----
        grad_from_prev : torch.Tensor
            Cumulative gradients from previous steps.

        Returns
        -------
        grad_to_next : torch.Tensor
            Cumulative gradients to next step.

        """
        # compute the gradient directly
        grad_to_bvec = torch.diagonal(grad_from_prev, 1)
        grad_to_dvec = torch.diagonal(grad_from_prev, -1)
        grad_to_E = _zero_tridiag(grad_from_prev) if ctx.noisy else None
        return grad_to_bvec, grad_to_dvec, grad_to_E


class BirthDeathShareVec(torch.autograd.Function):
    r"""Share a unique value inside a birth-death vector"""
    @staticmethod
    def forward(ctx, val, k, m=1):
        r"""Forwarding

        Args
        ----
        val : torch.Tensor
            Sharing value for vector.
        k : int
            Vector length
        m : int
            Maximum growing count.

        Returns
        -------
        vec : torch.Tensor
            Vector having the shared value.

        """
        # construct growing vector
        grow = torch.clamp(torch.arange(1, k + 1), max=m).type(val.dtype)

        # cache necessary attributes for backwarding
        ctx.grow = grow
        return grow * val

    @staticmethod
    def backward(ctx, grad_from_prev):
        r"""Backwarding

        Args
        ----
        grad_from_prev : torch.Tensor
            Cumulative gradients from previous steps.

        Returns
        -------
        grad_to_next : torch.Tensor
            Cumulative gradients to next step.

        """
        # compute the gradient directly
        grad_to_val = torch.sum(grad_from_prev * ctx.grow).view(-1)
        return grad_to_val, None, None


r"""
Rename
======
- **stdy_dist**    : **GenericSteadyDist**
- **bd_stdy_dist** : **BirthDeathSteadyDist**
- **bd_mat**       : **BirthDeathMat**
- **bd_shrvec**    : **BirthDeathShareVec**      
"""


# rename autograd functions
stdy_dist    = GenericSteadyDist.apply
bd_mat       = BirthDeathMat.apply
bd_shrvec    = BirthDeathShareVec.apply
mse_loss     = lambda h, y: torch.nn.functional.mse_loss(h, y, reduction='sum')


r"""
Model
========
- **MMmKModule**   : M/M/m/K Queue module
- **CondDistLossModule** : Conditional steady state distribution loss module
- **ResiDistLossModule** : Residual steady state distribution loss module
"""


class MMmKModule(torch.nn.Module):
    r"""M/M/m/K Queue Module"""
    def __init__(self, k, m=1, noise=True):
        r"""Initialize the class

        Args
        ----
        k : int
            Number of states. (Number of customers + 1.)
        m : int
            Number of servers.
        noise : bool
            Allow noise queue transaction.

        """
        # super calling
        torch.nn.Module.__init__(self)

        # save necessary attributes
        self.k = k
        self.m = m

        # explicitly allocate parameters
        self.mu = torch.nn.Parameter(torch.Tensor(1))
        if noise:
            self.E = torch.nn.Parameter(torch.Tensor(self.k, self.k))
        else:
            self.E = None

        # explicitly initialize parameters
        self.mu.data.fill_(1)
        if noise:
            self.E.data.fill_(0)
        else:
            pass

    def forward(self, lambd, ind=None):
        r"""Forwarding

        Args
        ----
        lambd : int
            Input lambda.
        ind : int or [int, ...]
            Focusing steady state indices.

        Returns
        -------
        dist : torch.Tensor
            Focusing steady state distribution.

        """
        # generate birth-death vector
        mus = bd_shrvec(self.mu, self.k - 1, self.m)
        lambds = bd_shrvec(lambd, self.k - 1)

        # get focusing steady states
        if self.E is None:
            return bd_stdy_dist(lambds, mus, ind)
        else:
            X = bd_mat(lambds, mus, self.E)
            return stdy_dist(X, ind)


class CondDistLossModule(torch.nn.Module):
    r"""Conditional Steady State Distribution Loss Module"""
    def forward(self, ind, output, pi, obvs):
        r"""Forwarding

        Args
        ----
        ind : torch.Tensor
            Indices to focus.
        output : torch.Tensor
            Output tensor.
        pi : torch.Tensor
            Target steady state distribution tensor.
        obvs : torch.Tensor
            Sampled observation tensor.

        Returns
        -------
        loss : torch.Tensor
            Loss scalor.

        """
        # preprocessing
        output = output[ind]
        target = obvs

        # assertion
        assert len(output.size()) == 1 and len(target.size()) == 1
        assert len(output) + 1 == len(target)

        # get conditional distribution and their observations
        cond_dist = output / output.sum()
        cond_obvs = target[0:-1]

        # compute loss
        loss = torch.sum(-cond_obvs * torch.log(cond_dist))
        return loss


class ResiDistLossModule(torch.nn.Module):
    r"""Residual Steady State Distribution Loss Module"""
    def forward(self, ind, output, pi, obvs):
        r"""Forwarding

        Args
        ----
        ind : torch.Tensor
            Indices to focus.
        output : torch.Tensor
            Output tensor.
        pi : torch.Tensor
            Target steady state distribution tensor.
        obvs : torch.Tensor
            Sampled observation tensor.

        Returns
        -------
        loss : torch.Tensor
            Loss scalor.

        """
        # preprocessing
        output = output[ind]
        target = obvs

        # assertion
        assert len(output.size()) == 1 and len(target.size()) == 1
        assert len(output) + 1 == len(target)

        # get conditional distribution and their observations
        cond_dist = torch.cat([output, 1 - output.sum().view(1)])
        cond_obvs = target

        # compute loss
        loss = torch.sum(-cond_obvs * torch.log(cond_dist))
        return loss


class DistMSELossModule(torch.nn.Module):
    r"""Steady State Distribution MSE Loss Module"""
    def forward(self, ind, output, pi, obvs):
        r"""Forwarding

        Args
        ----
        ind : torch.Tensor
            Indices to focus.
        output : torch.Tensor
            Output tensor.
        pi : torch.Tensor
            Target steady state distribution tensor.
        obvs : torch.Tensor
            Sampled observation tensor.

        Returns
        -------
        loss : torch.Tensor
            Loss scalor.

        """
        # preprocessing
        output = output[ind]
        target = pi[ind]
        
        # assertion
        assert len(output.size()) == 1 and len(target.size()) == 1

        # compute loss
        loss = mse_loss(output, target)
        return loss


r"""
Experiment
==========
- **WithRandom** : Class with Randomness Base
- **DataMMmK**   : Dataset for M/M/m/K Queue
- **TaskMMmK**   : Task for M/M/m/K Queue Model
"""


class WithRandom(object):
    r"""Class with Randomness Base"""
    def __init__(self, seed):
        r"""Initialize the class

        Args
        ----
        seed : int
            Random seed.

        """
        # configure random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


class DataMMmK(WithRandom):
    r"""Dataset for M/M/m/K Queue"""
    def __init__(self, n, k, m, const_mu, epsilon=1e-4, ind=[-3, -2], *args, **kargs):
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

        """
        # super call
        WithRandom.__init__(self, *args, **kargs)

        # save necessary attributes
        self.k = k
        self.m = m
        self._mu = torch.Tensor([const_mu])
        self._epsilon = epsilon
        self.ind = ind

        # create death vector
        self._mu_vec = bd_shrvec(self._mu, self.k - 1, self.m)

        # create noise transition
        self._noise = torch.Tensor(self.k, self.k)
        self._noise.normal_()
        self._noise = _zero_tridiag(self._noise)
        self._noise = self._noise / torch.norm(self._noise) * self._epsilon

        # generate samples
        self.samples = []
        lambd_cands = np.random.permutation(np.arange(1, const_mu * 2))[0:n]
        lambd_cands = sorted(lambd_cands)
        for const_lambd in lambd_cands:
            # generate birth-death variable vector
            _lambd = torch.Tensor([const_lambd])
            _lambd_vec = bd_shrvec(_lambd, self.k - 1)
        
            # generate ideal birth-death process
            X = bd_mat(_lambd_vec, self._mu_vec, self._noise)
            pi = stdy_dist(X)
            target = stdy_dist(X, self.ind)
            
            # sample observations from steady state on ideal birth-death process
            for i in range(5):
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

    def __repr__(self):
        r"""Get representation of the class

        Returns
        -------
        desc : str
            Representation of the class

        """
        # address representation items
        labels = ['Lambda', 'Pi', 'N']
        items, mxlen = [], [len(itr) for itr in labels]
        for itr in self.samples:
            item = []
            for i in range(3):
                item.append(re.sub(r'[\n\r] +', ' ', repr(itr[i].data.numpy())[6:-1]))
                mxlen[i] = max(mxlen[i], len(item[i]))
            items.append(item)

        # generate representation
        desc = "{0:{3}} {1:{4}} {2:{5}}\n".format(*labels, *mxlen)
        for item in items:
            line = "{0:{3}} {1:{4}} {2:{5}}\n".format(*item, *mxlen)
            desc = desc + line
        return desc


class TaskMMmK(WithRandom):
    r"""Task for M/M/m/k Queue Model"""
    # constants
    CTRL_CLS = dict(cond=CondDistLossModule, resi=ResiDistLossModule, mse=DistMSELossModule)
    OPTIM_CLS = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam, rms=torch.optim.RMSprop)

    def __init__(self, data, ctype, otype, ptype, lr=0.1, alpha=100, *args, **kargs):
        r"""Initialize the class

        Args
        ----
        data : object
            Dataset.
        ctype : str
            Criterion type.
        otype : str
            Optimizer type.
        ptype : str
            Propagation type.
        lr : float
            Learning rate.
        alpha : float
            Regularization strength.

        """
        # super call
        WithRandom.__init__(self, *args, **kargs)

        # save necessary attributes
        self.ctype = ctype
        self.ptype = ptype
        self.lr = lr
        self.alpha = alpha

        # allocate necessary attributes
        self.data = data
        self.layers = MMmKModule(k=self.data.k, m=self.data.m, noise=True)
        self.criterion = self.CTRL_CLS[ctype]()
        self.optimizer = self.OPTIM_CLS[otype](self.layers.parameters(), lr=self.lr)

    def fit_from_rand(self, num_epochs, color='green', name='task'):
        r"""Fit the model from randomness

        Args
        ----
        num_epochs : int
            Number of epochs to fit.
        color : str
            Visulization color.
        name : str
            Name prefix for saving files.

        """
        # set formatter
        fmt = "[{0:d}]\tTrain: {1:.6f}\tTest: {2:.6f}"

        # initialize buffer
        loss_tr = self.eval_train()
        loss_te = self.eval_test()
        self.loss_lst_tr, self.loss_lst_te = [loss_tr], [loss_te]
        print(fmt.format(0, loss_tr, loss_te))

        # fit parameters
        self.best_loss, self.best_params = None, None
        for epc in range(1, num_epochs + 1):
            try:
                # shuffle data
                dind = np.random.permutation(len(self.data))
        
                # train
                getattr(self, "_{}_ffbp".format(self.ptype))(dind)
        
                # evaluate
                loss_tr = self.eval_train()
                loss_te = self.eval_test()
                self.loss_lst_tr.append(loss_tr)
                self.loss_lst_te.append(loss_te)
                print(fmt.format(epc, loss_tr, loss_te))
    
                # update best parameters
                if self.best_loss is None or loss_te < self.best_loss:
                    self.best_loss = loss_te
                    self.best_params = copy.deepcopy(self.layers.state_dict())
                else:
                    pass
            except Exception as err:
                print(err)
                break
        torch.save((self.loss_lst_tr, self.loss_lst_te), "{}_loss_lst.pt".format(name))

        # get ideal result
        self.ideal_loss_tr = self.eval_train(ideal=True)
        self.ideal_loss_te = self.eval_test(ideal=True)
        torch.save((self.ideal_loss_tr, self.ideal_loss_te), "{}_ideal_loss.pt".format(name))

    def _single_ffbp(self, ind):
        r"""Single-batch Forwarding and Backwarding

        Args
        ----
        ind : numpy.ndarray
            Permuted indices to forward.

        """
        # forward and backward
        for idx in ind:
            self.optimizer.zero_grad()
            lambd, pi, obvs = self.data.samples[idx]
            output = self.layers.forward(lambd)
            loss = self.criterion.forward(self.data.ind, output, pi, obvs)
            loss = loss + self.alpha * torch.norm(self.layers.E)
            loss.backward()
            self.optimizer.step()

    def _full_ffbp(self, ind):
        r"""Full-batch Forwarding and Backwarding

        Args
        ----
        ind : numpy.ndarray
            Permuted indices to forward.

        """
        # forward and backward
        self.optimizer.zero_grad()
        loss_lst = []
        for idx in ind:
            lambd, pi, obvs = self.data.samples[idx]
            output = self.layers.forward(lambd)
            loss = self.criterion.forward(self.data.ind, output, pi, obvs)
            loss_lst.append(loss)
        loss = sum(loss_lst) + self.alpha * torch.norm(self.layers.E)
        loss.backward()
        self.optimizer.step()

    def eval_train(self, ideal=False):
        r"""Evaluate training cases
        
        Args
        ----
        ideal : bool
            Evaluate ideal case.

        """
        # forward
        loss_lst = []
        for i in range(len(self.data)):
            lambd, pi, obvs = self.data.samples[i]
            output = pi if ideal else self.layers.forward(lambd)
            loss = self.criterion.forward(self.data.ind, output, pi, obvs)
            loss_lst.append(loss)
        loss = sum(loss_lst) / len(loss_lst)
        return loss.data.item()

    def eval_test(self, ideal=False):
        r"""Evaluate test cases
        
        Args
        ----
        ideal : bool
            Evaluate ideal case.

        """
        # forward
        loss_lst = []
        for i in range(len(self.data)):
            lambd, pi, obvs = self.data.samples[i]
            output = pi if ideal else self.layers.forward(lambd)
            loss = mse_loss(output[-1], pi[-1])
            loss_lst.append(loss)
        loss = sum(loss_lst) / len(loss_lst)
        return loss.data.item()


r"""
Study
=====
- **MM1K** : M/M/1/K Hyper Parameter Study
"""


def MM1K(data_seed=47, model_seed=47):
    r"""M/M/1/K Hyper Parameter Tuning

    Args
    ----
    data_seed : int
        Random seed for data generation.
    model_seed : int
        Random seed for model initializaion and tuning.

    """
    # generate data
    data = DataMMmK(100, k=6, m=1, const_mu=25, epsilon=1e-4, ind=[-3, -2], seed=data_seed)
    print("#Data: {}".format(len(data)))

    # traverse loss and batch settings
    ctype_lst  = ['mse', 'cond', 'resi']
    btype_lst  = ['single'] # // ['single', 'full']
    otype_lst  = ['adam']   # // ['sgd', 'adam', 'rms']
    lr_str_lst = ['1e-2']   # // ['1e-1', '1e-2', '1e-3']
    alpha_lst  = [1000]     # // [1, 10, 100, 1000]
    hyper_combs = itertools.product(ctype_lst, btype_lst, otype_lst, lr_str_lst, alpha_lst)
    num_epochs  = 100
    for combine in hyper_combs:
        ctype, btype, otype, lr_str, alpha = combine
        name = "{}_{}_{}_{}_{}".format(ctype, btype, otype, lr_str, alpha)
        print("==[{}]==".format(name))
        task = TaskMMmK(data, ctype, otype, btype, lr=float(lr_str), alpha=alpha, seed=model_seed)
        task.fit_from_rand(num_epochs, name=name)


if __name__ == '__main__':
    r"""Main Entrance"""
    # set precision
    np.set_printoptions(precision=8, suppress=True)
    torch.Tensor = torch.DoubleTensor

    # do targeting hyper study
    MM1K()