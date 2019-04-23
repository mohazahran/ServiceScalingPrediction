'''
Created on Apr 22, 2019

@author: Mohamed Zahran
'''
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim
import random
import copy
import sys
import math
import solver
import numpy as np
import scipy.stats
import matplotlib as mpl
from Logger import Logger

# mpl.use('TkAgg')
import matplotlib.pyplot as plt


class GenericQueue_customGradients(nn.Module):
    def __init__(self, params):
        super(GenericQueue_customGradients, self).__init__()

        self.params = params
        self.K = params['K']
        self.c = params['c']
        self.paramCount = 0
        self.mus = nn.ParameterDict()
        self.paramId_2_muId = {}
        cntr = 0
        if params['modelType'] == 'MMmK':
            self.m = float(params['m'])
            self.mus['0'] = nn.Parameter(torch.DoubleTensor([params['initialMu']]), requires_grad=True)
            self.paramId_2_muId[cntr] = '0'

        elif params['modelType'] == 'muPerState':  # MMmK with different mu per state
            for i in range(self.params['K'] - 1):
                # self.mus[str(i)] = nn.Parameter(torch.DoubleTensor([params['initialMu']*(i+1)]), requires_grad=True)
                self.mus[str(i)] = nn.Parameter(torch.DoubleTensor([params['initialMu']]), requires_grad=True)
                self.paramId_2_muId[cntr] = str(i)
                cntr += 1


        elif params['modelType'] == 'embeddedMC':  # each state has a different mu for all previous states
            for i in range(1, self.params['K']):
                for j in range(i):
                    self.mus[str(i) + ',' + str(j)] = nn.Parameter(torch.DoubleTensor([params['initialMu']]),
                                                                   requires_grad=True)
                    self.paramId_2_muId[cntr] = str(i) + ',' + str(j)
                    cntr += 1

        self.optimiser = getattr(torch.optim, params['optim'])(self.parameters(), lr=params['lr'])
        if self.params['use_lr_scheduler']:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=[50, 100, 250, 500],
                                                                  gamma=0.5)

        paramsCount = 0
        for p in list(self.parameters()):
            if len(p.shape) == 1:
                paramsCount += p.shape[0]
            else:
                paramsCount += p.shape[0] * p.shape[1]

        print '**************************************************'
        print 'Number of parameters in the model = ', paramsCount
        self.paramCount = paramsCount

    def mus_2_str(self):
        str_mus = ''
        for i in range(self.paramCount):
            str_mus += 'mu%d=%f ' % (i, self.mus[self.paramId_2_muId[i]].item())
        return str_mus

    def paramClip(self):
        for i in range(self.paramCount):
            self.mus[self.paramId_2_muId[i]].data.clamp_(min=1.0, max=1e20)

    def form_Q(self, inputLambda):
        if self.params['modelType'] == 'muPerState':
            if self.params['cuda'] == True:
                self.Q = torch.zeros((self.params['K'], self.params['K'])).cuda()
            else:
                self.Q = torch.zeros((self.params['K'], self.params['K']))
            for i in range(self.params['K']):
                for j in range(self.params['K']):
                    if i == 0 and j == 0:
                        self.Q[i][j] = -inputLambda
                    elif i == j - 1:
                        self.Q[i][j] = inputLambda
                    elif i - 1 == j:
                        self.Q[i][j] = self.mus[str(i - 1)]
                    elif i == j:
                        if i == self.params['K'] - 1 and j == self.params['K'] - 1:
                            self.Q[i][j] = -1 * self.mus[str(i - 1)]

                        else:
                            self.Q[i][j] = -1 * (self.mus[str(i - 1)] + inputLambda)
                    else:
                        self.Q[i][j] = 0.0

        elif self.params['modelType'] == 'MMmK':
            self.Q = torch.zeros((self.K, self.K))
            for i in range(self.K):
                for j in range(self.K):
                    if i == 0 and j == 0:
                        self.Q[i][j] = -inputLambda
                    elif i == j - 1:
                        self.Q[i][j] = inputLambda
                    elif i - 1 == j:
                        if i < self.m:
                            self.Q[i][j] = self.mus['0'] * i
                        else:
                            self.Q[i][j] = self.mus['0'] * self.m
                    elif i == j:
                        if i == self.K - 1 and j == self.K - 1:
                            if i < self.m:
                                self.Q[i][j] = -1 * (i) * self.mus['0']
                            else:
                                self.Q[i][j] = -1 * (self.m) * self.mus['0']
                        else:
                            if i < self.m:
                                tmp = -1 * (self.mus['0'] * i + inputLambda)
                            else:
                                tmp = -1 * (self.mus['0'] * self.m + inputLambda)
                            self.Q[i][j] = tmp
                    else:
                        self.Q[i][j] = 0.0

        if self.params['modelType'] == 'embeddedMC':
            if self.params['cuda'] == True:
                self.Q = torch.zeros((self.params['K'], self.params['K'])).cuda()
            else:
                self.Q = torch.zeros((self.params['K'], self.params['K']))

            self.Q[0][0] = -inputLambda
            self.Q[0][1] = inputLambda
            for i in range(1, self.params['K']):
                for j in range(i):
                    self.Q[i][j] = self.mus[str(i) + ',' + str(j)]
                if i + 1 < self.params['K']:
                    self.Q[i][i + 1] = inputLambda
                self.Q[i][i] = -1 * torch.sum(self.Q[i])

        return self.Q

    def form_dQ_dmu(self):
        if self.params['modelType'] == 'MMmK':
            dQ = torch.zeros((self.K, self.K))
            for i in range(1, self.K):
                if i < self.m:
                    dQ[i, i - 1] = i
                    dQ[i, i] = -i
                else:
                    dQ[i, i - 1] = self.m
                    dQ[i, i] = -self.m
            return dQ
        elif self.params['modelType'] == 'muPerState':
            dQ_dmus = torch.zeros((self.paramCount, self.K, self.K))
            for i in range(self.paramCount):
                dQ_dmus[i, i + 1, i] = 1.0
                dQ_dmus[i, i + 1, i + 1] = -1.0
            return dQ_dmus

    def E_geometric(self, geo_p, P, dP_dmus, P_inf):
        x = scipy.stats.geom.rvs(geo_p)
        Pr_x = scipy.stats.geom.pmf(x, geo_p)
        P_inv = torch.inverse(P)
        P_inv_power_neg_i = P_inv
        P_power_i_1 = torch.diag(torch.ones(self.params['K']))

        expectn = torch.zeros((self.paramCount, self.K, self.K), dtype=torch.double)
        for i in range(1, x + 1):
            ex = torch.matmul(P_inv_power_neg_i, dP_dmus)
            ex = torch.matmul(P_power_i_1, ex)
            one_cdf_i = scipy.stats.geom.sf(i, geo_p)
            ex = torch.div(ex, one_cdf_i)
            expectn = expectn + ex
            if i + 1 < x + 1:
                P_power_i_1 = torch.mm(P_power_i_1, P)
                P_inv_power_neg_i = torch.mm(P_inv_power_neg_i, P_inv)

        expectn = torch.mul(expectn, Pr_x)
        limit_t_inf = torch.matmul(P_inf, expectn)
        return limit_t_inf

    def E_geometric_infSplit(self, geo_p, P, dP_dmus, P_inf):
        # geo_p = self.params['geo_p']
        x = scipy.stats.geom.rvs(geo_p)
        Pr_x = scipy.stats.geom.pmf(x, geo_p)
        P_power_i_1 = torch.diag(torch.ones(self.params['K']))

        expectn = torch.zeros((self.K, self.K))
        for i in range(1, x + 1):
            one_cdf_i = scipy.stats.geom.sf(i, geo_p)
            expectn = expectn + torch.div(P_power_i_1, one_cdf_i)
            if i + 1 < x + 1:
                P_power_i_1 = torch.mm(P_power_i_1, P)

        part1 = torch.matmul(P_inf, dP_dmus)
        part1 = torch.matmul(part1, expectn)

        part2 = torch.matmul(expectn, dP_dmus)
        part2 = torch.matmul(part2, P_inf)

        limit_t_inf = part1 + part2
        return limit_t_inf

    def calGradient_without_russian_roulette(self, P, dP_dmus, t=2):
        # t = 2^2^c --> log(t) = 2^c log(2) --> log(t)/log(2) = 2^c --> log(log(t)/log(2)) = c log(2)
        # --> c = log(log(t)/log(2)) / log(2)
        res = torch.zeros((self.paramCount, self.K, self.K))
        P_pow_k_1 = torch.diag(torch.ones(self.params['K']))
        P_pow_t_k = torch.diag(torch.ones(self.params['K']))
        for k in range(1, t + 1):
            pow = t - k
            # c = math.floor(math.log(pow) / math.log(2))
            if pow == 0:
                P_pow_t_k = torch.diag(torch.ones(self.params['K']))
            elif pow == 1:
                P_pow_t_k = P
            else:
                c = math.floor(math.log(math.log(pow) / math.log(2)) / math.log(2))
                if c > 0:
                    currPow = 2 ** (2 ** (c))
                    P_pow_t_k = P
                    for j in range(int(c + 1)):
                        P_pow_t_k = torch.mm(P_pow_t_k, P_pow_t_k)
                else:
                    currPow = 0
                for j in range(int(pow - currPow)):
                    P_pow_t_k = torch.mm(P_pow_t_k, P)

            tmp = torch.matmul(P_pow_t_k, dP_dmus)
            res = res + torch.matmul(tmp, P_pow_k_1)

            P_pow_k_1 = torch.mm(P_pow_k_1, P)
        return res

    def no_russian_roulette_infSplit(self, P, dP_dmus, P_inf):
        I = torch.diag(torch.ones(self.params['K']))
        inv = torch.inverse(I - P)
        part1 = torch.matmul(P_inf, dP_dmus)
        part1 = torch.matmul(part1, inv)

        part2 = torch.matmul(inv, dP_dmus)
        part2 = torch.matmul(part2, P_inf)

        return part1 + part2

    def matrixPower(self, P, t):
        c = math.floor(math.log(math.log(t) / math.log(2)) / math.log(2))
        if c > 0:
            currPow = 2 ** (2 ** (c))
            Pt = P
            for j in range(int(c + 1)):
                Pt = torch.mm(Pt, Pt)
        else:
            currPow = 0
            Pt = P
        for j in range(int(t - currPow)):
            Pt = torch.mm(Pt, P)
        return Pt

    def objective(self, inputLa, dropCount):
        '''

        :param inputLa: the current input rate
        :param dropCount: the correspondng number of drops
        :return: loss, gradient (dLoss/dmus)

        dL(g(P^t))/dmus = dL(g(P^t))/g(P^t) * dg(P^t)/P^t * dP^t/dQ

        '''


        #1) Forming Q
        #=======================
        Q = self.form_Q(inputLa)

        #2) Forming P
        #=======================
        if self.params['cuda'] == True:
            I = torch.diag(torch.ones(self.params['K'])).cuda()
            gamma = torch.max(I * torch.abs(self.Q)) + self.c
            P = I + torch.div(self.Q, gamma).cuda()
        else:
            I = torch.diag(torch.ones(self.params['K']))
            gamma = torch.max(I * torch.abs(self.Q)) + self.c
            P = I + torch.div(self.Q, gamma)

        #3) Calculating PK
        #=======================

        #baseline P^t
        if self.params['russian_roulette'] == False and self.params['inf_split'] == False:
            P_t = self.matrixPower(P, self.params['no_russian_roulette_t'])
            PK = sum(P_t[:, -1]) / float(self.K)
        else: #Infinity learning
            # calculating pi
            Pprime = P.data.numpy() - np.eye(self.K, self.K)
            Pprime = np.concatenate((Pprime, np.ones((self.K, 1))), axis=1)
            b = np.zeros((1, self.K + 1))
            b[-1][-1] = 1
            pi, residues, rank, s = np.linalg.lstsq(Pprime.T, b.T, rcond=None)

            #populating P^inf
            P_inf = torch.DoubleTensor(np.repeat(pi.T, self.K, axis=0))

            #getting PK = g(P^inf)
            PK = torch.DoubleTensor([pi[-1][-1]])
            PK = np.clip(PK, 1e-20, 1.0)

        #4) calculting dL(g(P^inf))/g(P^inf)
        #====================================
        dNLL = (-dropCount / PK) + (inputLa - dropCount) / (1 - PK)
        dNLL = dNLL.double()

        #5) calculating dg(P^t)/P^t
        #============================
        dg_dH = torch.zeros((self.K, self.K))
        # dg_dH[-1][-1] = 1.0
        dg_dH[:, self.K - 1] = 1.0 / self.K


        #6) calculting dP^inf/dmus
        #===========================
        dP_dmus = torch.zeros((self.paramCount, self.K, self.K))
        if self.params['modelType'] == 'MMmK':
            dQ_dmu = self.form_dQ_dmu()
            dgamma_dmu = self.m
            dP_dmus[0] = torch.div((torch.mul(dQ_dmu, gamma) - torch.mul(Q, dgamma_dmu)), (gamma ** 2))
        elif self.params['modelType'] == 'muPerState':
            dQ_dmus = self.form_dQ_dmu()
            maxEntry = torch.argmax(I * torch.abs(self.Q))
            maxMuId = (maxEntry.item() // self.K) - 1
            for i in range(self.paramCount):
                if maxMuId == i:
                    dgamma_dmu = 1.0
                else:
                    dgamma_dmu = 0.0

                dP_dmus[i] = torch.div((torch.mul(dQ_dmus[i], gamma) - torch.mul(Q, dgamma_dmu)), (gamma ** 2))
        else:
            pass

        if self.params['russian_roulette'] == True:
            if self.params['inf_split']:
                limit_t_inf = self.E_geometric_infSplit(self.params['geo_p'], P, dP_dmus, P_inf)
            else:
                limit_t_inf = self.E_geometric(self.params['geo_p'], P, dP_dmus, P_inf)
        elif self.params['russian_roulette'] == False:
            if self.params['inf_split']:
                limit_t_inf = self.no_russian_roulette_infSplit(P, dP_dmus, P_inf)
            else:
                limit_t_inf = self.calGradient_without_russian_roulette(P, dP_dmus,
                                                                        t=self.params['no_russian_roulette_t'])
        #7) calculating dL(g(P^inf))/dmus
        #==================================
        dLoss_dmus = dNLL.item() * torch.sum(dg_dH * limit_t_inf, (1, 2))

        try:
            loss = (-1 * dropCount * math.log(PK)) - ((inputLa - dropCount) * math.log(1 - PK))
        except:
            print PK
            exit(1)

        return loss, torch.log(PK), dLoss_dmus

    def predict(self, inputLa, clampNumbers=False):
        Q = self.form_Q(inputLa)
        # print self.mu, Q
        if self.params['cuda'] == True:
            I = torch.diag(torch.ones(self.params['K'])).cuda()
            gamma = torch.max(I * torch.abs(self.Q)) + self.c
            P = torch.diag(torch.ones(self.params['K'])).cuda() + torch.div(self.Q, gamma).cuda()
        else:
            I = torch.diag(torch.ones(self.params['K']))
            gamma = torch.max(I * torch.abs(self.Q)) + self.c
            P = torch.diag(torch.ones(self.params['K'])) + torch.div(self.Q, gamma)

        Pprime = P.data.numpy() - np.eye(self.K, self.K)
        Pprime = np.concatenate((Pprime, np.ones((self.K, 1))), axis=1)
        b = np.zeros((1, self.K + 1))
        b[-1][-1] = 1

        pi, residues, rank, s = np.linalg.lstsq(Pprime.T, b.T, rcond=None)
        return pi[-1][-1]


############################################################################################################

def evaluate_given_model(testModel, testLambdas, testPKs, useDropRate=True, plotFig=True):
    testModel.eval()
    est_PKs = []
    avg_NLL = 0.0

    for i in range(len(testLambdas)):
        est_PK = testModel.predict(testLambdas[i])
        logPK = math.log(np.clip(est_PK, 1e-10, 1.0))
        # logPK = math.log(est_PK)
        est_PKs.append(est_PK)
        if not useDropRate:
            actualDrops = float(math.ceil(testPKs[i] * testLambdas[i]))
        else:
            actualDrops = testPKs[i]
        NLL = - (actualDrops * logPK + (testLambdas[i] - actualDrops) * math.log(1 - math.exp(logPK)))
        avg_NLL += NLL

    if testModel.params['cuda'] == True:
        est_PK_torch = torch.DoubleTensor(est_PKs).cuda()
        testPKs_torch = torch.DoubleTensor(testPKs).cuda()
    else:
        est_PK_torch = torch.DoubleTensor(est_PKs)
        testPKs_torch = torch.DoubleTensor(testPKs)

    MSE = F.mse_loss(est_PK_torch, testPKs_torch)
    # avg_NLL = avg_NLL / float(len(testLambdas))
    print
    'Evaluation: MSE=', MSE.item(), 'NLL=', avg_NLL

    if plotFig:
        fig = plt.figure(1, figsize=(6, 4))
        axes = plt.gca()
        ax = plt.axes()

        # drawing est PK vs. emp PK
        plt.ylabel('Probability')
        plt.xlabel('Lambda')
        lines = plt.plot(testLambdas, est_PKs, '--r', label='Estimated PK')
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(testLambdas, testPKs, 'b', label='True PK')
        plt.setp(lines, linewidth=2.0)
        plt.legend(loc=2, prop={'size': 17}, labelspacing=0.1)
        fig.suptitle('Estimated PK Vs. True PK', fontsize=12, fontweight='bold', horizontalalignment='center', y=.86)
        plt.grid()
        plt.show()

    testModel.train()
    return MSE.item(), avg_NLL


def train(model=None,
          inputLambdas=[], PKs=[],
          batchSize=10,
          epochs=150,
          validLambdas=[], validPKs=[],
          shuffleData=True,
          showBatches=False,
          useDropRate=True
          ):
    model.train()
    bestValidLoss = 1e10
    bestModel = model

    if model.params['cuda'] == True:
        inputLambdas = Variable(torch.DoubleTensor(inputLambdas)).cuda()
        PKs = Variable(torch.DoubleTensor(PKs)).cuda()

        validLambdas = Variable(torch.DoubleTensor(validLambdas)).cuda()
        validPKs = Variable(torch.DoubleTensor(validPKs)).cuda()
    else:
        inputLambdas = Variable(torch.DoubleTensor(inputLambdas))
        PKs = Variable(torch.DoubleTensor(PKs))

        validLambdas = Variable(torch.DoubleTensor(validLambdas))
        validPKs = Variable(torch.DoubleTensor(validPKs))

    total_loss = 0.0
    for e in range(epochs):
        if model.params['use_lr_scheduler']:
            model.scheduler.step()
        # shuffle training data
        if shuffleData:
            rows = len(inputLambdas)
            idxs = list(range(0, rows))
            random.shuffle(idxs)
            idxs = torch.LongTensor(idxs)
            train_X = inputLambdas[idxs]
            train_Y = PKs[idxs]

        epochLoss = 0.0
        est_PKs = []
        true_PKs = []
        gradient = 0.0
        for b in range(len(inputLambdas)):
            inputLa = inputLambdas[b]
            if not useDropRate:
                dropProb = PKs[b]
                dropCount = float(math.ceil(dropProb * inputLa))
            else:
                dropCount = PKs[b]

            loss, logPK, dLoss_dmu = model.objective(inputLa, dropCount)

            gradient += dLoss_dmu.data.numpy()

            est_PKs.append(torch.exp(logPK).item())
            true_PKs.append(PKs[b])


            total_loss = total_loss + loss
            epochLoss += loss.item()

            if (b) % batchSize == 0:
                total_loss = total_loss / float(batchSize)
                gradient /= float(batchSize)
                gradient = np.clip(gradient, -1 * model.params['gradientClip'], model.params['gradientClip'])
                for p in range(model.paramCount):
                    model.mus[model.paramId_2_muId[p]].grad = torch.DoubleTensor([gradient[p]])
                model.optimiser.step()
                for group in model.optimiser.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.detach()
                            p.grad.zero_()
                model.paramClip()

                if showBatches:
                    print 'batch#', b, 'loss=', total_loss.item(), '\n', model.mus_2_str(), '\ngrad=', gradient
                gradient = 0.0
                total_loss = 0.0

        total_loss = 0.0
        est_PKs = torch.DoubleTensor(est_PKs)
        true_PKs = torch.DoubleTensor(true_PKs)
        MSE = F.mse_loss(est_PKs, true_PKs)

        validMSE, validNLL = evaluate_given_model(model, validLambdas, validPKs, useDropRate, plotFig=False)
        validLoss = validNLL / float(len(validLambdas))

        model.validLoss = validLoss
        model.trainLoss = epochLoss / float(len(inputLambdas))

        print 'Epoch ', e, '--------------------------------------'
        for param_group in model.optimiser.param_groups:
            print 'lr=', param_group['lr'],
        print 'TrainLoss= ', epochLoss
        #print 'MSE=', MSE.item()
        print 'validLoss=', validLoss
        print 'mus=', model.mus_2_str()

        if validLoss < bestValidLoss:
            bestValidLoss = validLoss
            torch.save(model, model.params['modelName'])
            bestModel = torch.load(model.params['modelName'])
        else:
            pass
            # model.params['lr'] /= 2.0

        print '\tbest validLoss=', bestModel.validLoss.item(), 'best trainLoss=', bestModel.trainLoss
        print '\tbest mu=', bestModel.mus_2_str()
        print

        torch.save(bestModel, bestModel.params['modelName'])


def run_using_MMmK_simulation():
    mpl.rcParams.update({'font.size': 17})
    random.seed(1111)


    true_m = 50
    true_K = 100
    true_mu = 25.0
    singleValue = 1175
    sampleCount = 10

    #simulating MMmK
    train_X = [singleValue] * sampleCount
    PKs = [math.exp(solver.M_M_m_K_log(float(inp) / float(true_mu), true_m, true_K)) for inp in train_X]
    train_Y = [np.random.poisson(lam=train_X[i] * PKs[i], size=1)[0] for i in range(len(train_X))]

    print 'min,max,avg,tot #drops=', min(train_Y), max(train_Y), np.mean(train_Y), sum(train_Y)
    valid_X = train_X
    valid_Y = train_Y
    ########################################

    params = \
        {
            'cuda': False,
            'm': true_m,
            'K': true_K,
            'c': 0.001,  # uniformization const.
            'geo_p': 0.1,
            'inf_split': True,
            'russian_roulette': False,
            'no_russian_roulette_t': 100, #this is the 't' in P^t (the baseline)
            'initialMu': 15.0,
            'modelType': 'MMmK',  # 'MMmK', 'muPerState' (BirthDeath)
            'modelName': 'simData1',
            'optim': 'Adam',
            'lr': 0.05,
            'use_lr_scheduler': False,
            'gradientClip': 5.0  # for exploding gradients
        }

    print 'Parameters:\n', params

    if torch.cuda.is_available():
        print 'torch.cuda.is_available() = ', torch.cuda.is_available()

    random.seed(1111)
    if params['cuda'] == True:
        torch.cuda.manual_seed(1111)
    else:
        torch.manual_seed(1111)

    model = GenericQueue_customGradients(params)
    if params['cuda'] == True:
        model.cuda()
    else:
        torch.set_num_threads(4)

    train(model=model,
          inputLambdas=train_X, PKs=train_Y,
          batchSize=1,
          epochs=10000,
          validLambdas=valid_X, validPKs=valid_Y,
          shuffleData=True,
          showBatches=False,
          useDropRate=True
          )

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    np.random.seed(1111)
    random.seed(1111)
    run_using_MMmK_simulation()
    print('DONE!')