import numpy as np
import scipy.stats as stats
import torch


r"""
Utils
=====
- **_numpy_solve**                : Solve linear system by numpy
- **_steady_dist**                : Get steady state distributuion
- **_russian_roulette**           : Infinite product summation with Russian Roulette
- **_inf_split**                  : Infinite product summation with infinite split
- **_russian_roulette_inf_split** : infinite product summation with Russian Roulette and infinite split
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
    b = torch.zeros(k + 1, 1, dtype=A.dtype, device=A.device)
    b[-1] = 1
    pi = solver(A, b).t()
    return pi


def _russian_roulette(P, Pinf, M, dist, *args, **kargs):
    r"""Infinite product summation without Russian Roulette

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

    # compute all necessary power to save time
    t = max(x, RR_T)
    prod_lst = [torch.eye(k, dtype=P.dtype, device=P.device)]
    for i in range(t - 1):
        prod_lst.append(torch.matmul(P, prod_lst[-1]))

    # compute expectation with infinity split
    E = torch.zeros(k, k, dtype=P.dtype, device=P.device)
    for i in range(1, t + 1):
        cdf_above = stats.geom.sf(i, *args, **kargs)
        part1 = prod_lst[t - i]
        part2 = prod_lst[i - 1]
        E = E + torch.div(torch.matmul(torch.matmul(part1, M), part2), cdf_above)
    return E


def _inf_split(P, Pinf, M, dist, *args, **kargs):
    r"""Infinite product summation with infinite split trick

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

    # compute expectation with infinity split
    I = torch.eye(k, dtype=P.dtype, device=P.device)
    try:
        inv = torch.inverse(I - P)
    except:
        inv = torch.zeros(k, k, dtype=P.dtype, device=P.device)
    part1 = torch.matmul(torch.matmul(Pinf, M), inv)
    part2 = torch.matmul(torch.matmul(inv, M), Pinf)
    return part1 + part2


def _russian_roulette_inf_split(P, Pinf, M, dist, *args, **kargs):
    r"""Infinite product summation with Russian Roulette and infinite split trick

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
    return part1


r"""
Autograd
========
- **GenericSteadyDist** : Differentiable steady state distribution function for generic queue process
"""


class GenericSteadyDist(torch.autograd.Function):
    r"""Differentiable steady state distribution function for generic queue process"""
    # constants
    GEO_P = 0.1
    RR = None

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
        cls = GenericSteadyDist

        # get P
        P, Q, (gamma, _id) = cls._X2P(k, X, c, dtype=X.dtype, device=X.device)

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
    def _X2P(k, X, c, dtype, device):
        r"""Construct selection gradient factor

        Args
        ----
        k : int
            Number of transition states.
        X : torch.Tensor
            Tensor X.
        c : float
            Uniform normalization offset.
        dtype : torch.dtype
            Tensor dtype.
        device : torch.device
            Tensor dtype.

        Returns
        ------
        P : torch.Tensor
            Tensor P.
        Q : torch.Tensor
            Tensor Q.
        gamma : float
            Uniform normalization factor.
        _id : int
            Row ID which gives gamma.

        """
        # get diagonal line for uniform normalization
        diags = torch.sum(X, dim=1)
        diags_mx = torch.diagflat(diags)

        # get gamma for uniform normalization
        gamma, _id = torch.max(diags), torch.argmax(diags)
        gamma, _id = gamma.item() + c, _id.item()

        # get P
        Q = X - diags_mx
        P = torch.eye(k, dtype=Q.dtype, device=Q.device) + Q / gamma
        return P, Q, (gamma, _id)

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
        coeff_S = GenericSteadyDist.RR(P, Pinf, coeff_P, stats.geom, geo_p)
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


r"""
Function
========
- **stdy_dist_rrx** : Russian Roulette Steady State Distribution Function
- **stdy_dist_pow** : Power Method Steady State Distribution Function
- **stdy_dist**     : Steady State Distribution Function Interface
"""


def stdy_dist_rrx(X, ind=None, c=0.001, vmin=1e-20, vmax=1, geo_p=0.1, trick='hav', *args, **kargs):
    r"""Steady state distribution by Russian Roulette

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
    geo_p : float
        Geometry distrbution parameters.
    trick : str
        Raussian Roulette trick to use.

    Returns
    -------
    pi : torch.Tensor
        Steady state distribution vector.

    """
    # assertion
    assert len(X.size()) == 2
    assert len(torch.nonzero(torch.diagonal(X))) == 0

    # claim trick dict
    TRICK_DICT = {
        'rr'   : _russian_roulette,
        'inf'  : _inf_split,
        'rrinf': _russian_roulette_inf_split,
    }

    # adjust Russian Roulette trick
    global RR_T
    GenericSteadyDist.GEO_P = geo_p
    GenericSteadyDist.RR    = TRICK_DICT[trick]
    RR_T                    = 10

    # call autograd function class
    return GenericSteadyDist.apply(X, ind, c, vmin, vmax, *args, **kargs)


def stdy_dist_pow(X, ind=None, c=0.001, vmin=1e-20, vmax=1, *args, **kargs):
    r"""Steady state distribution by power method

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

    Returns
    -------
    pi : torch.Tensor
        Steady state distribution vector.

    """
    # assertion
    assert len(X.size()) == 2
    assert len(torch.nonzero(torch.diagonal(X))) == 0

    # adjust Russian Roulette trick
    global RR_T
    GenericSteadyDist.GEO_P = None
    GenericSteadyDist.RR    = None
    RR_T                    = None

    # get necessary attributes
    k = X.size(1)
    device = X.device
    ind = [ind] if isinstance(ind, int) else (ind or list(range(k)))
    cls = GenericSteadyDist

    # get P
    P, Q, (gamma, _id) = cls._X2P(k, X, c, dtype=X.dtype, device=X.device)

    # use power P^{2^{z}} to get pi by linear system
    z = 7
    E = P
    for i in range(z):
        E = torch.matmul(E, E)
    pi = torch.mean(E, dim=0, keepdim=True)
    pi = torch.clamp(pi, vmin, vmax).squeeze()

    # return selected distribution
    return pi[ind]


def stdy_dist(method, *args, **kargs):
    r"""Steady state distribution interface

    Args
    ----
    method : str
        Method used to get steady state.

    Returns
    -------
    pi : torch.Tensor
        Steady state distribution vector.

    """
    # cliam method dict
    METHOD_DICT = {
        'rrx': stdy_dist_rrx,
        'pow': stdy_dist_pow,
    }
    pi = METHOD_DICT[method](*args, **kargs)
    return pi