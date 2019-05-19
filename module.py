import numpy as np
import torch
import time
import os
import copy
import traceback
import function as F


r"""
Module
======
- **QueueModule**        : Queue module without priori
- **MMmKModule**         : M/M/m/K Queue module
- **LBWBModule**         : Leaky Bucket Web Browsing Queue module
- **CIOModule**          : Circular Input/Output Queue module
- **CondDistLossModule** : Conditional steady state distribution loss module
- **ResiDistLossModule** : Residual steady state distribution loss module
- **MSEDistLossModule**  : MSE steady state distribution loss module
"""


class QueueModule(torch.nn.Module):
    r"""Queue Module without Priori"""
    def __init__(self, prior, method, **kargs):
        r"""Initialize the class

        Args
        ----
        prior : data.QueueData
            Queue prior holder.
        method : str
            Steady state distribution method.

        """
        # super calling
        torch.nn.Module.__init__(self)

        # save necessary attributes
        self.prior = prior
        self.method = method
        self.kargs = kargs
        self.k = self.prior.k

        # explicitly allocate parameters
        self.X = torch.nn.Parameter(torch.Tensor(self.k, self.k))

        # explicitly initialize parameters
        self.X.data.fill_(0)

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
        # ensure valid parameters
        self.zero_biz(self.X)

        # generate matrix
        X = self.prior.update_input_prior(self.X, lambd)
        return F.stdy_dist(self.method, X, ind, **self.kargs)

    def zero_biz(self, *args):
        r"""Force all bizarre values in the argument to be zero"""
        # traverse argument
        for mx in args:
            # ensure non-NaN
            mx.data[mx != mx] = 0
            
            # ensure non-negative
            mx.data[mx < 0] = 0


class MMmKModule(QueueModule):
    r"""M/M/m/K Queue Module"""
    def __init__(self, prior, method, **kargs):
        r"""Initialize the class

        Args
        ----
        prior : data.QueueData
            Queue prior holder.
        method : str
            Steady state distribution method.

        """
        # super calling
        torch.nn.Module.__init__(self)

        # save necessary attributes
        self.prior = prior
        self.method = method
        self.kargs = kargs
        self.k = self.prior.k

        # explicitly allocate parameters
        self.mu = torch.nn.Parameter(torch.Tensor(1))
        self.E = torch.nn.Parameter(torch.Tensor(self.k, self.k))
        self.mxargs = (self.mu,)

        # explicitly initialize parameters
        self.mu.data.fill_(1)
        self.E.data.fill_(0)

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
        # ensure valid parameters
        self.zero_biz(self.E, *self.mxargs)

        # generate matrix
        self.E = self.prior.zero_prior(self.E)
        X = self.prior.mx(self.E, lambd, *self.mxargs)
        return F.stdy_dist(self.method, X, ind, **self.kargs)


class LBWBModule(QueueModule):
    r"""Leaky Bucket Web Browsing Queue Module"""
    def __init__(self, prior, method, **kargs):
        r"""Initialize the class

        Args
        ----
        prior : data.QueueData
            Queue prior holder.
        method : str
            Steady state distribution method.

        """
        # super calling
        torch.nn.Module.__init__(self)

        # save necessary attributes
        self.prior = prior
        self.method = method
        self.kargs = kargs
        self.k = self.prior.k

        # explicitly allocate parameters
        self.mu = torch.nn.Parameter(torch.Tensor(1))
        self.bucket = torch.nn.Parameter(torch.Tensor(1))
        self.E = torch.nn.Parameter(torch.Tensor(self.k, self.k))
        self.mxargs = (self.mu, self.bucket)

        # explicitly initialize parameters
        self.mu.data.fill_(1)
        self.bucket.data.fill_(1)
        self.E.data.fill_(0)

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
        # ensure valid parameters
        self.zero_biz(self.E, *self.mxargs)

        # generate matrix
        self.E = self.prior.zero_prior(self.E)
        X = self.prior.mx(self.E, lambd, *self.mxargs)
        return F.stdy_dist(self.method, X, ind, **self.kargs)


class CIOModule(QueueModule):
    r"""Circular Input/Output Queue Module"""
    def __init__(self, prior, method, **kargs):
        r"""Initialize the class

        Args
        ----
        prior : data.QueueData
            Queue prior holder.
        method : str
            Steady state distribution method.

        """
        # super calling
        torch.nn.Module.__init__(self)

        # save necessary attributes
        self.prior = prior
        self.method = method
        self.kargs = kargs
        self.k = self.prior.k

        # explicitly allocate parameters
        self.mu = torch.nn.Parameter(torch.Tensor(1))
        self.E = torch.nn.Parameter(torch.Tensor(self.k, self.k))
        self.mxargs = (self.mu,)

        # explicitly initialize parameters
        self.mu.data.fill_(1)
        self.E.data.fill_(0)

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
        # ensure valid parameters
        self.zero_biz(self.E, *self.mxargs)

        # generate matrix
        self.E = self.prior.zero_prior(self.E)
        X = self.prior.mx(self.E, lambd, *self.mxargs)
        return F.stdy_dist(self.method, X, ind, **self.kargs)


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
        cond_dist = torch.clamp(cond_dist, 1e-20, 1)
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
        cond_dist = torch.clamp(cond_dist, 1e-20, 1)
        cond_obvs = target

        # compute loss
        global temp
        loss = torch.sum(-cond_obvs * torch.log(cond_dist))
        return loss


class MSEDistLossModule(torch.nn.Module):
    r"""MSE Steady State Distribution Loss Module"""
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
        loss = torch.nn.functional.mse_loss(output, target, reduction='sum')
        return loss


r"""
Experiment
==========
- **Task** : Task for All Above Queue Model
"""


class Task(object):
    r"""Task for All Above Queue Model"""
    # constants
    CTRL_CLS = dict(cond=CondDistLossModule, resi=ResiDistLossModule, mse=MSEDistLossModule)

    def __init__(self, train_data, test_data, layers, ctype, alpha, seed):
        r"""Initialize the class

        Args
        ----
        train_data : object
            Train dataset.
        test_data : object
            Test dataset.
        layers : torch.nn.Module
            Neural network module to tune.
        ctype : str
            Criterion type.
        alpha : float
            Regularization strength.
        seed : int
            Random seed.

        """
        # configure random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # save necessary attributes
        self.ctype = ctype
        self.ptype = 'single'
        self.lr = 0.01
        self.alpha = alpha

        # allocate necessary attributes
        self.train_data = train_data
        self.test_data = test_data
        self.layers = layers
        self.criterion = self.CTRL_CLS[ctype]()
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=self.lr)

    def fit_from_rand(self, num_epochs, root='.', name='task'):
        r"""Fit the model from randomness

        Args
        ----
        num_epochs : int
            Number of epochs to fit.
        root : str
            Folder to save results.
        name : str
            Name prefix for saving files.

        """
        # output configuration
        print('\033[31;1m', end='')
        print(self.train_data.__class__.__name__)
        print(self.layers.__class__.__name__)
        print(self.ctype)
        print(self.ptype)
        print(self.alpha)
        print('\033[0m', end='')

        # set formatter
        fmt = "[{0:d}]\tTrain: {1:.6f}\tTest: {2:.6f}\tParams: {3}"

        # get ideal result
        self.ideal_loss_tr = self.eval_train(ideal=True)
        self.ideal_loss_te = self.eval_test(ideal=True)

        # initialize buffer
        loss_tr = self.eval_train()
        loss_te = self.eval_test()
        param_rec = copy.deepcopy(self.layers.state_dict())
        del param_rec['E']
        param_rec = {key: val.data.item() for key, val in param_rec.items()}
        print(fmt.format(0, loss_tr, loss_te, param_rec))
        self.loss_lst_tr, self.loss_lst_te = [loss_tr], [loss_te]
        self.param_lst = [param_rec]
        self.time_lst = [0]

        # fit parameters
        self.best_loss, self.best_params = None, None
        for epc in range(1, num_epochs + 1):
            try:
                # shuffle data
                dind = np.random.permutation(len(self.train_data))

                # train
                timer = time.time()
                getattr(self, "_{}_ffbp".format(self.ptype))(dind)
                time_cost = time.time() - timer

                # evaluate
                loss_tr = self.eval_train()
                loss_te = self.eval_test()
                param_rec = copy.deepcopy(self.layers.state_dict())
                del param_rec['E']
                param_rec = {key: val.data.item() for key, val in param_rec.items()}
                print(fmt.format(epc, loss_tr, loss_te, param_rec))
                if np.isnan(loss_tr) or np.isnan(loss_te):
                    print('force to stop by nan')
                    break
                else:
                    self.loss_lst_tr.append(loss_tr)
                    self.loss_lst_te.append(loss_te)
                    self.param_lst.append(param_rec)
                    self.time_lst.append(time_cost)

                # update best parameters
                if self.best_loss is None or loss_te < self.best_loss:
                    self.best_loss = loss_te
                    self.best_params = copy.deepcopy(self.layers.state_dict())
                else:
                    pass
            except Exception as err:
                msg = traceback.format_exc()
                print("force to stop by <{}>".format(msg))
                break

        # save result
        save_dict = {
            'loss_lst_tr'  : self.loss_lst_tr,
            'loss_lst_te'  : self.loss_lst_te,
            'ideal_loss_tr': self.ideal_loss_tr,
            'ideal_loss_te': self.ideal_loss_te,
            'param'        : self.best_params,
            'ideal_param'  : self.test_data.param_dict(),
        }
        torch.save(save_dict, os.path.join(root, "{}.pt".format(name)))

        # output some convergence information
        fmt1 = "loss: {:.6f} ({:.6f})"
        fmt2 = "[{}]: {:.3f} ({:.3f})"
        fmt3 = "[{}]: {} ({:.6f})"
        print(fmt1.format(min(self.loss_lst_te), self.ideal_loss_te))
        for name, true_param in self.test_data.param_dict().items():
            try:
                pred_param = self.layers.state_dict()[name].data.item()
                print(fmt2.format(name, float(pred_param), float(true_param)))
            except:
                print(fmt3.format(name, 'generic', float(true_param)))

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
            lambd, pi, obvs = self.train_data.samples[idx]
            output = self.layers.forward(lambd)
            loss = self.criterion.forward(self.train_data.ind, output, pi, obvs)
            if self.alpha > 0:
                loss = loss + self.alpha * torch.norm(self.layers.E)
            else:
                pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.layers.parameters(), 5)
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
        for i in range(len(self.train_data)):
            lambd, pi, obvs = self.train_data.samples[i]
            output = pi if ideal else self.layers.forward(lambd)
            loss = self.criterion.forward(self.train_data.ind, output, pi, obvs)
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
        for i in range(len(self.test_data)):
            lambd, pi, obvs = self.test_data.samples[i]
            focus = self.test_data.focus
            output = pi if ideal else self.layers.forward(lambd)
            loss = torch.nn.functional.mse_loss(output[focus], pi[focus], reduction='sum')
            loss_lst.append(loss)
        loss = sum(loss_lst) / len(loss_lst)
        return loss.data.item()