'''
Created on Oct 27, 2018

@author: mohame11
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

#mpl.use('TkAgg')
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

        elif params['modelType'] == 'muPerState': # MMmK with different mu per state
            for i in range(self.params['K'] - 1):
                #self.mus[str(i)] = nn.Parameter(torch.DoubleTensor([params['initialMu']*(i+1)]), requires_grad=True)
                self.mus[str(i)] = nn.Parameter(torch.DoubleTensor([params['initialMu']]), requires_grad=True)
                self.paramId_2_muId[cntr] = str(i)
                cntr += 1


        elif params['modelType'] == 'embeddedMC': # each state has a different mu for all previous states
            for i in range(1, self.params['K']):
                for j in range(i):
                    self.mus[str(i)+','+str(j)] = nn.Parameter(torch.DoubleTensor([params['initialMu']]), requires_grad=True)
                    self.paramId_2_muId[cntr] = str(i)+','+str(j)
                    cntr += 1

        self.optimiser = getattr(torch.optim, params['optim'])(self.parameters(), lr=params['lr'])
        if self.params['use_lr_scheduler']:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=[50,100,250,500], gamma=0.5)

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
            str_mus += 'mu%d=%f '%(i,self.mus[self.paramId_2_muId[i]].item())
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
                    self.Q[i][j] = self.mus[str(i)+','+str(j)]
                if i+1 < self.params['K']:
                    self.Q[i][i+1] = inputLambda
                self.Q[i][i] = -1*torch.sum(self.Q[i])

        return self.Q

    def form_dQ_dmu(self):
        '''
        dQ = np.zeros((self.K, self.K))
        for i in range(1, self.K):
            if i < self.m:
                dQ[i,i-1] = i
                dQ[i,i] = -i
            else:
                dQ[i,i-1] = self.m
                dQ[i,i] = -self.m
        return dQ
        '''
        if self.params['modelType'] == 'MMmK':
            dQ = torch.zeros((self.K, self.K))
            for i in range(1, self.K):
                if i < self.m:
                    dQ[i,i-1] = i
                    dQ[i,i] = -i
                else:
                    dQ[i,i-1] = self.m
                    dQ[i,i] = -self.m
            return dQ
        elif self.params['modelType'] == 'muPerState':
            dQ_dmus = torch.zeros((self.paramCount, self.K, self.K))
            for i in range(self.paramCount):
                dQ_dmus[i,i+1,i] = 1.0
                dQ_dmus[i,i+1,i+1] = -1.0
            return dQ_dmus

    def form_dP_dmu2(self, P):
        #from torch.autograd import grad
        #dfdx, = grad(f(x_), x_, only_inputs=True, create_graph=True, retain_graph=False)
        #torch.autograd.grad(P, self.mus)
        #torch.autograd.grad(P, self.mus['0'], only_inputs=True, create_graph=True, retain_graph=True)
        #https://discuss.pytorch.org/t/autograd-grad-accumulates-gradients-on-sequence-of-tensor-making-it-hard-to-calculate-hessian-matrix/18032/2
        dP_dmus = torch.zeros((self.paramCount, self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                try:
                    grads = torch.autograd.grad(P[i][j], self.mus.values(), only_inputs=True, retain_graph=True)
                    for k in range(self.paramCount):
                        dP_dmus[k][i][j] = grads[k]
                        #self.mus[str(k)].grad.zero_()
                except:
                    pass
        return dP_dmus

    def form_dP_dmu(self, P):
        #from torch.autograd import grad
        #dfdx, = grad(f(x_), x_, only_inputs=True, create_graph=True, retain_graph=False)
        #torch.autograd.grad(P, self.mus)
        #torch.autograd.grad(P, self.mus['0'], only_inputs=True, create_graph=True, retain_graph=True)
        dP_dmus = torch.zeros((self.paramCount, self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                try:
                    #P[i][j].backward(create_graph = True)
                    P[i][j].backward(retain_graph = True)
                    for k in range(self.paramCount):
                        dP_dmus[k][i][j] = self.mus[self.paramId_2_muId[k]].grad
                        self.mus[self.paramId_2_muId[k]].grad.zero_()
                except:
                    pass
        return dP_dmus

    def E_geometric(self, geo_p, P, dP_dmus, P_inf):
        x = scipy.stats.geom.rvs(geo_p)
        Pr_x = scipy.stats.geom.pmf(x, geo_p)
        P_inv = torch.inverse(P)
        P_inv_power_neg_i = P_inv
        P_power_i_1 = torch.diag(torch.ones(self.params['K']))

        expectn = torch.zeros((self.paramCount, self.K, self.K), dtype = torch.double)
        for i in range(1, x + 1):
            ex = torch.matmul(P_inv_power_neg_i, dP_dmus)
            ex = torch.matmul(P_power_i_1, ex)
            one_cdf_i = scipy.stats.geom.sf(i, geo_p)
            ex = torch.div(ex, one_cdf_i)
            expectn = expectn + ex
            if i+1 < x+1:
                P_power_i_1 = torch.mm(P_power_i_1, P)
                P_inv_power_neg_i = torch.mm(P_inv_power_neg_i, P_inv)

        expectn = torch.mul(expectn, Pr_x)
        limit_t_inf = torch.matmul(P_inf, expectn)
        return limit_t_inf

    def E_geometric_infSplit(self, geo_p, P, dP_dmus, P_inf):
        #geo_p = self.params['geo_p']
        x = scipy.stats.geom.rvs(geo_p)
        Pr_x = scipy.stats.geom.pmf(x, geo_p)
        P_power_i_1 = torch.diag(torch.ones(self.params['K']))
        
        expectn = torch.zeros((self.K, self.K))
        for i in range(1, x + 1):
            one_cdf_i = scipy.stats.geom.sf(i, geo_p)
            expectn = expectn + torch.div(P_power_i_1, one_cdf_i)
            if i+1 < x+1:
                P_power_i_1 = torch.mm(P_power_i_1, P)

        part1 = torch.matmul(P_inf, dP_dmus)
        part1 = torch.matmul(part1, expectn)

        part2 = torch.matmul(expectn, dP_dmus)
        part2 = torch.matmul(part2, P_inf)

        limit_t_inf = part1 + part2
        return limit_t_inf
    
    def calGradient_without_russian_roulette(self, P, dP_dmus, t=2):
        #t = 2^2^c --> log(t) = 2^c log(2) --> log(t)/log(2) = 2^c --> log(log(t)/log(2)) = c log(2)
        #--> c = log(log(t)/log(2)) / log(2)
        res = torch.zeros((self.paramCount, self.K, self.K))
        P_pow_k_1 = torch.diag(torch.ones(self.params['K']))
        P_pow_t_k = torch.diag(torch.ones(self.params['K']))
        for k in range(1,t+1):
            pow = t-k
            #c = math.floor(math.log(pow) / math.log(2))
            if pow == 0:
                P_pow_t_k = torch.diag(torch.ones(self.params['K']))
            elif pow == 1:
                P_pow_t_k = P
            else:
                c = math.floor ( math.log(math.log(pow)/math.log(2)) / math.log(2) )
                if c > 0:
                    currPow = 2**(2**(c))
                    P_pow_t_k = P
                    for j in range(int(c+1)):
                        P_pow_t_k = torch.mm(P_pow_t_k,P_pow_t_k)
                else:
                    currPow = 0
                for j in range(int(pow-currPow)):
                    P_pow_t_k = torch.mm(P_pow_t_k, P)
            
            tmp = torch.matmul(P_pow_t_k, dP_dmus)
            res = res + torch.matmul(tmp, P_pow_k_1)
            
                
            P_pow_k_1 = torch.mm(P_pow_k_1, P) 
        return res

    def matrixPower(self, P, t):
        c = math.floor( math.log(math.log(t)/math.log(2)) / math.log(2) )
        if c > 0:
            currPow = 2**(2**(c))
            Pt = P
            for j in range(int(c+1)):
                Pt = torch.mm(Pt, Pt)
        else:
            currPow = 0
            Pt = P
        for j in range(int(t-currPow)):
            Pt = torch.mm(Pt, P)
        return Pt

    def objective(self, inputLa, dropCount):
        Q = self.form_Q(inputLa)
        if self.params['cuda'] == True:
            I = torch.diag(torch.ones(self.params['K'])).cuda()
            gamma = torch.max(I * torch.abs(self.Q)) + self.c
            P = I + torch.div(self.Q, gamma).cuda()
        else:
            I = torch.diag(torch.ones(self.params['K']))
            gamma = torch.max(I * torch.abs(self.Q)) + self.c
            P = I + torch.div(self.Q, gamma)
            
        
        if self.params['russian_roulette'] == False:
            #P_t = P
            #for i in range(self.params['no_russian_roulette_t']):
            #   P_t = torch.mm(P_t, P)
            P_t = self.matrixPower(P, self.params['no_russian_roulette_t'])
            PK = sum(P_t[:,-1])/float(self.K)
        else:
            
            Pprime = P.data.numpy() - np.eye(self.K, self.K)
            Pprime = np.concatenate((Pprime,np.ones((self.K,1))), axis = 1)
            b = np.zeros((1, self.K+1))
            b[-1][-1] = 1
            pi, residues, rank, s = np.linalg.lstsq(Pprime.T, b.T, rcond=None)
    
            P_inf = torch.DoubleTensor(np.repeat(pi.T, self.K, axis=0))
    
            PK = torch.DoubleTensor([pi[-1][-1]])  #g(H)
            PK = np.clip(PK, 1e-20, 1.0)

        dNLL = (-dropCount/PK) + (inputLa-dropCount)/(1-PK)
        dNLL = dNLL.double()

        dg_dH = torch.zeros((self.K, self.K))
        #dg_dH[-1][-1] = 1.0
        dg_dH[:,self.K-1] = 1.0/self.K
        
        
        dP_dmus = torch.zeros((self.paramCount, self.K, self.K))
        if self.params['modelType'] == 'MMmK':
            dQ_dmu = self.form_dQ_dmu()
            dgamma_dmu = self.m
            dP_dmus[0] = torch.div( (torch.mul(dQ_dmu, gamma) - torch.mul(Q, dgamma_dmu)) , (gamma ** 2) )
        elif self.params['modelType'] == 'muPerState':
            dQ_dmus = self.form_dQ_dmu()
            maxEntry = torch.argmax(I * torch.abs(self.Q))
            maxMuId = (maxEntry.item() // self.K)-1
            for i in range(self.paramCount):
                if maxMuId == i:
                    dgamma_dmu = 1.0
                else:
                    dgamma_dmu = 0.0
                    
                dP_dmus[i] = torch.div( (torch.mul(dQ_dmus[i], gamma) - torch.mul(Q, dgamma_dmu)) , (gamma ** 2) )
        else:
            pass
        
        
        #auto diff
        #dP_dmus = self.form_dP_dmu(P)
        
        '''
        iters = 50
        grads = []
        for j in range(iters):
            E = self.E_geometric(self.params['geo_p'], P, dP_dmu)
            dLoss_dmu = dNLL.item() * torch.sum(dg_dH * E).item()
            grads.append(dLoss_dmu)
        avgGrad = sum(grads)/float(len(grads))
        varGrad = sum([(g-avgGrad)**2 for g in grads])/float(len(grads))
        print 'mu=%f Geo_p=%f avgGrad=%f var=%f' % (self.mu.item(), self.params['geo_p'], avgGrad, varGrad) 
        #print self.mu.item(), self.params['geo_p'], avgGrad, varGrad
        '''
        
        if self.params['russian_roulette'] == False:
            limit_t_inf = self.calGradient_without_russian_roulette(P, dP_dmus, t=self.params['no_russian_roulette_t'])
        else:
            if self.params['inf_split']:
                limit_t_inf = self.E_geometric_infSplit(self.params['geo_p'], P, dP_dmus, P_inf)
            else:
                limit_t_inf = self.E_geometric(self.params['geo_p'], P, dP_dmus, P_inf)

        dLoss_dmus = dNLL.item() * torch.sum(dg_dH * limit_t_inf, (1,2))

        try:
            loss = (-1 * dropCount * math.log(PK)) - ((inputLa - dropCount) * math.log(1 - PK))
        except:
            print PK
            exit(1)

        return loss, torch.log(PK), dLoss_dmus





    def predict(self, inputLa, clampNumbers=False):
        Q = self.form_Q(inputLa)
        #print self.mu, Q
        if self.params['cuda'] == True:
            I = torch.diag(torch.ones(self.params['K'])).cuda()
            gamma = torch.max(I * torch.abs(self.Q)) + self.c
            P = torch.diag(torch.ones(self.params['K'])).cuda() + torch.div(self.Q, gamma).cuda()
        else:
            I = torch.diag(torch.ones(self.params['K']))
            gamma = torch.max(I * torch.abs(self.Q)) + self.c
            P = torch.diag(torch.ones(self.params['K'])) + torch.div(self.Q, gamma)

        Pprime = P.data.numpy() - np.eye(self.K, self.K)
        Pprime = np.concatenate((Pprime,np.ones((self.K,1))), axis = 1)
        b = np.zeros((1, self.K+1))
        b[-1][-1] = 1

        pi, residues, rank, s = np.linalg.lstsq(Pprime.T, b.T, rcond=None)
        return pi[-1][-1]

############################################################################################################


def parseDataFile(fpath, inputPacketsCols, droppedPacketsCols):
    df = pd.read_csv(fpath, usecols=inputPacketsCols + droppedPacketsCols)
    df.fillna(0, inplace=True)  # replace missing values (NaN) to zero9
    return df


def getTrainingData(dir, summaryFile, minDropRate, maxDropRate, useDropRate):
    sfile = dir + summaryFile
    inputPacketsCols = ['CallRate(P)']
    droppedPacketsCols = ['FailedCall(P)']

    df = pd.read_csv(sfile, usecols=['Rate File', ' Failed Calls'])
    df.fillna(0, inplace=True)
    train_X = []
    train_Y = []
    for i, row in df.iterrows():
        if row[' Failed Calls'] < minDropRate or row[' Failed Calls'] > maxDropRate:
            continue
        fname = 'sipp_data_' + row['Rate File'] + '_1.csv'
        if fname == 'sipp_data_long_var_rate_0_1836_seconds_1.csv':
            continue
        simulationFile = dir + fname  # sipp_data_UFF_Perdue_01_1_reduced_1.csv     UFF_Perdue_01_12_reduced

        curr_df = pd.read_csv(simulationFile, usecols=inputPacketsCols + droppedPacketsCols)
        curr_df.fillna(0, inplace=True)  # replace missing values (NaN) to zero9
        for j, curr_row in curr_df.iterrows():
            try:
                the_lambda = float(curr_row['CallRate(P)'])
                failed = float(curr_row['FailedCall(P)'])
                if failed > the_lambda or the_lambda <= 0:
                    continue
                if useDropRate:
                    PK = failed
                else:
                    PK = failed / the_lambda
            except:
                continue
            train_X.append(Variable(torch.DoubleTensor([the_lambda])))
            train_Y.append(Variable(torch.DoubleTensor([PK])))

    return train_X, train_Y

def getTrainingData_withFilter(dir, summaryFile, minDropRate, maxDropRate, useDropRate):
    sfile = dir + summaryFile
    inputPacketsCols = ['CallRate(P)']
    droppedPacketsCols = ['FailedCall(P)']

    df = pd.read_csv(sfile, usecols=['Rate File', ' Failed Calls'])
    df.fillna(0, inplace=True)
    train_X = []
    train_Y = []
    for i, row in df.iterrows():
        if row[' Failed Calls'] < 1:
            continue
        fname = 'sipp_data_' + row['Rate File'] + '_1.csv'
        if fname == 'sipp_data_long_var_rate_0_1836_seconds_1.csv':
            continue
        simulationFile = dir + fname  # sipp_data_UFF_Perdue_01_1_reduced_1.csv     UFF_Perdue_01_12_reduced

        curr_df = pd.read_csv(simulationFile, usecols=inputPacketsCols + droppedPacketsCols)
        curr_df.fillna(0, inplace=True)  # replace missing values (NaN) to zero9
        for j, curr_row in curr_df.iterrows():
            try:
                the_lambda = float(curr_row['CallRate(P)'])
                failed = float(curr_row['FailedCall(P)'])
                if failed > the_lambda or the_lambda <= 0:
                    continue
                if useDropRate:
                    PK = failed
                else:
                    PK = failed / the_lambda
            except:
                continue
            train_X.append(Variable(torch.DoubleTensor([the_lambda])))
            train_Y.append(Variable(torch.DoubleTensor([PK])))
    
    combined = list(zip(train_X, train_Y))
    random.shuffle(combined)
    train_X[:], train_Y[:] = zip(*combined)
    
    currDrops = 0
    X = []
    Y = []
    it = 0
    while currDrops < maxDropRate:
        currDrops += train_Y[it].item()
        X.append(train_X[it])
        Y.append(train_Y[it])
        it += 1
        
    

    return X, Y


def evaluate_given_model(testModel, testLambdas, testPKs, useDropRate = True, plotFig=True):
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
    #avg_NLL = avg_NLL / float(len(testLambdas))
    print 'Evaluation: MSE=', MSE.item(), 'NLL=', avg_NLL

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

            #if b == 100:
            #    exit(1)

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

            #print(est_PKs[-1], true_PKs[-1].item()),

            total_loss = total_loss + loss
            epochLoss += loss.item()

            if (b) % batchSize == 0:
                total_loss = total_loss / float(batchSize)
                gradient /= float(batchSize)
                gradient = np.clip(gradient, -1*model.params['gradientClip'], model.params['gradientClip'])
                for p in range(model.paramCount):
                    model.mus[model.paramId_2_muId[p]].grad = torch.DoubleTensor([gradient[p]])
                    #model.mus[model.paramId_2_muId[p]].grad = torch.FloatTensor([gradient[p]])
                #model.mu.data = model.mu.data - model.params['lr'] * gradient
                #model.mu.data.clamp_(min = 1.0, max = 1e10)
                #print 'new mu=', model.mu.item()
                #print model.mu, gradient
                model.optimiser.step()
                #model.mu.data.clamp_(min = 1.0, max = 1e10)
                for group in model.optimiser.param_groups:
                    for p in group['params']:
                       if p.grad is not None:
                           p.grad.detach()
                           p.grad.zero_()

                #model.optimiser.zero_grad()
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
        #validMSE, validNLL = 0, 0
        #if model.params['dataLossType'] == 'MSE':
        #    validLoss = validMSE
        #elif model.params['dataLossType'] == 'NLL':
        #    validLoss = validNLL
        validLoss = validNLL / float(len(validLambdas))

        model.validLoss = validLoss
        model.trainLoss = epochLoss / float(len(inputLambdas))



        print 'Epoch ',e, '--------------------------------------'
        for param_group in model.optimiser.param_groups:
            print 'lr=', param_group['lr'],
        print 'TrainLoss= ', epochLoss
        print 'MSE=', MSE.item()
        print 'validLoss=', validLoss
        print 'mus=',model.mus_2_str()


        if validLoss < bestValidLoss:
            bestValidLoss = validLoss
            torch.save(model, model.params['modelName'])
            bestModel = torch.load(model.params['modelName'])
        else:
            pass
            #model.params['lr'] /= 2.0

        print '\tbest validLoss=', bestModel.validLoss.item(), 'best trainLoss=', bestModel.trainLoss
        print '\tbest mu=',bestModel.mus_2_str()
        print

        torch.save(bestModel, bestModel.params['modelName'])





def run_using_MMmK_simulation():
    mpl.rcParams.update({'font.size': 17})
    random.seed(1111)
    
    '''
    true_m = 4
    true_K = 250
    true_mu = 25.0 
    singleValue = 125   
    sampleCount = 50
    '''
    
    '''
    true_m = 3
    true_K = 5
    true_mu = 50.0 
    singleValue = 45
    
    sampleCount = 50
    '''
    
    
    true_m = 50
    true_K = 100
    true_mu = 25.0 
    singleValue = 1175
    sampleCount = 10
    
    
    
    '''
    true_m = 5
    true_K = 25
    true_mu = 100.0
    singleValue = 75
    sampleCount = 50
    '''
    
    '''
    true_mus = [i+2 for i in range(24)]
    true_K = 25
    true_mu = 100.0
    true_m = -1
    singleValue = 25
    sampleCount = 50
    '''


    #train_X = list(range(1, maxLambda, 10))
    #random.shuffle(train_X)
    train_X = [singleValue]*sampleCount
    #train_X = [singleValue]+[i for i in range(2,50)]
    
    PKs = [math.exp(solver.M_M_m_K_log(float(inp) / float(true_mu), true_m, true_K)) for inp in train_X]
    #PKs = [solver.bith_death_closedForm([float(inp)]*(true_K-1), true_mus, true_K-1) for inp in train_X]

    #drops = [math.ceil(train_X[i] * train_Y[i]) for i in range(len(train_X))]
    train_Y = [np.random.poisson(lam=train_X[i] * PKs[i], size=1)[0] for i in range(len(train_X))]
    #drops = [math.ceil(train_X[i] * train_Y[i]) for i in range(len(train_X))]

    print 'min,max,avg,tot #drops=', min(train_Y), max(train_Y), np.mean(train_Y), sum(train_Y)
    #exit(1)
    #valid_X = list(range(1, maxLambda, 7))
    # valid_X = list(range(5, 1000, 7))
    #random.shuffle(valid_X)
    #valid_X = [singleValue]    
    valid_X = train_X
    #valid_X = valid_X[:inputDataSize]
    valid_Y = train_Y

    ########################################

    params = \
    {
        'cuda': False,
        'm': true_m,
        'K': true_K,
        'c': 0.001,     #uniformization const.
        'geo_p': 0.1,
        'inf_split': False,
        'russian_roulette': True,
        'no_russian_roulette_t': 100,
        'initialMu': 15.0,
        'modelType': 'MMmK',  # 'MMmK', 'muPerState', 'embeddedMC'
        'modelName': 'simData1',
        'optim': 'Adam',
        'lr': 0.05  ,
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
    
    '''
    model = torch.load('MMmK_simData1')
    model.params['lr'] = 1.0
    for param_group in model.optimiser.param_groups:
        param_group['lr'] = model.params['lr']
    #model.scheduler.last_epoch = 0
    #print 'last_epoch', model.scheduler.last_epoch
     #58609   
    model.train()
    prevMus = 'best mu= mu0=1.000000 mu1=1.000000 mu2=1.000000 mu3=1.000000 mu4=1.000000 mu5=1.000000 mu6=1.000000 mu7=1.000000 mu8=1.000000 mu9=1.000000 mu10=1.000000 mu11=1.000000 mu12=1.000000 mu13=1.000000 mu14=1.000000 mu15=1.000000 mu16=1.000000 mu17=1.000000 mu18=1.000000 mu19=1.000000 mu20=1.000000 mu21=1.000000 mu22=1.000000 mu23=1.000000 mu24=1.000000 mu25=1.000000 mu26=1.000000 mu27=1.000000 mu28=1.000000 mu29=1.000000 mu30=1.000000 mu31=1.000000 mu32=1.000000 mu33=1.000000 mu34=1.000000 mu35=1.000000 mu36=1.000000 mu37=1.000000 mu38=1.000000 mu39=1.000000 mu40=1.000000 mu41=1.000000 mu42=1.000000 mu43=1.000000 mu44=1.000000 mu45=1.000000 mu46=1.000000 mu47=1.000000 mu48=1.000000 mu49=1.000000 mu50=1.000000 mu51=1.000000 mu52=1.000000 mu53=1.000000 mu54=1.000000 mu55=1.000000 mu56=1.000000 mu57=1.000000 mu58=1.000000 mu59=1.000000 mu60=1.000000 mu61=1.000000 mu62=1.000000 mu63=1.000000 mu64=1.000000 mu65=1.000000 mu66=1.000000 mu67=1.000000 mu68=1.000000 mu69=1.000000 mu70=1.000000 mu71=1.000000 mu72=1.000000 mu73=1.000000 mu74=1.000000 mu75=1.000000 mu76=1.000000 mu77=1.000000 mu78=1.000000 mu79=1.000000 mu80=1.000000 mu81=1.000000 mu82=1.000000 mu83=1.000000 mu84=1.000000 mu85=1.000483 mu86=2.181110 mu87=11.519135 mu88=49.047550 mu89=86.010762 mu90=124.655592 mu91=171.366547 mu92=242.353646 mu93=347.272606 mu94=491.947895 mu95=721.457692 mu96=1062.043319 mu97=1564.547120 mu98=2363.274530'
    spltMus = prevMus.split()[2:]
    for m in range(len(spltMus)):
        parts = spltMus[m].split('=')
        model.mus[parts[0][-1]].data = torch.DoubleTensor([float(parts[-1])])
    
    #model.mus['0'].data = torch.DoubleTensor([265.519831])
    '''
    

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
          useDropRate = True
          )

    testLambdas = list(range(1, maxLambda, 1))
    # testLambdas = list(range(15, 1000, 10))
    testPKs = [math.exp(solver.M_M_m_K_log(float(inp) / float(true_mu), true_m, true_K)) for inp in testLambdas]
    evaluate_given_model(model, testLambdas, testPKs)


def run_using_real_data():
    random.seed(1111)

    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_INVITE_CORE_1_K_425982_SCALE_60_REMOTE_CPU_2019.01.08-01.56.08/sipp_results/'
    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_4_K_DEFT_SCALE_86_2018.10.29-22.19.41/sipp_results/'
    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_2_K_100000_SCALE_43_2018.11.03-13.38.21/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.4+.5_CORE_1_K_425982_SCALE_24_2018.11.23-18.47.32/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_INVITE_CORE_1_K_425982_SCALE_17_2018.11.21-02.49.02/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_MSG_CORE_1_K_425982_SCALE_80_2018.11.21-04.56.39/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_SUBSCRIBE_CORE_1_K_425982_SCALE_36_2018.11.21-07.04.10/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.5+0_CORE_1_K_425982_SCALE_30_2018.11.23-20.55.11/sipp_results/'

    summaryFile = 'summary_data_dump.csv'

    trainQuota = 0.75
    validQuota = 0.25
    useDropRate = True

    data_X, data_Y = getTrainingData_withFilter(dir, summaryFile, minDropRate=1, maxDropRate=2500, useDropRate = True)

    # shuffle data
    combined = list(zip(data_X, data_Y))
    random.shuffle(combined)
    data_X[:], data_Y[:] = zip(*combined)

    trainLen = int(trainQuota * len(data_X))
    validLen = int(validQuota * len(data_X))

    train_X = data_X[:trainLen]
    train_Y = data_Y[:trainLen]

    valid_X = data_X[trainLen:trainLen + validLen]
    valid_Y = data_Y[trainLen:trainLen + validLen]

    test_X = data_X[trainLen + validLen:]
    test_Y = data_Y[trainLen + validLen:]
    
    print 'trainLen=', trainLen, 'validLen=', validLen
    print '#train drops=', sum(train_Y), '#valid drops=', sum(valid_Y)

    ########################################
    params = {
        'cuda': False,
        'm': 3,
        'K': 5,
        'c': 0.001,           #uniformization const.
        'geo_p': 0.5,
        'inf_split': False,
        'initialMu': 1000.0,
        'modelType': 'MMmK',  # 'MMmK', 'muPerState', 'embeddedMC'
        'modelName': 'realData2_muPerState_lowDrops2500_K5',
        'optim': 'Adam',
        'lr': 1.0,
        'use_lr_scheduler': True,
        'gradientClip': 25.0  # for exploding gradients
    }
    
    sys.stdout = Logger()
    logPath = params['modelName']+'_log'
    sys.stdout.log = open(logPath, "a")
    print 'Parameters:\n', params
    
    # print torch.__version__

    # print torch.version.cuda

    if torch.cuda.is_available():
        print 'torch.cuda.is_available() = ', torch.cuda.is_available()

    if params['cuda'] == True:
        torch.cuda.manual_seed(1111)
    else:
        torch.manual_seed(1111)

    model = GenericQueue_customGradients(params)
    
    #model = torch.load('muPerState')
    #model.params['lr'] = 10.0
    #model.train()

    if params['cuda'] == True:
        model.cuda()
    else:
        torch.set_num_threads(4)

    train(model=model,
          inputLambdas=train_X, PKs=train_Y,
          batchSize=32,
          epochs=5000,
          validLambdas=valid_X, validPKs=valid_Y,
          shuffleData=True,
          showBatches=False,
          useDropRate = True
          )


def test():
    mpl.rcParams.update({'font.size': 17})
    random.seed(1111)
   
    true_m = 50
    true_K = 100
    true_mu = 25.0    
    sampleCount = 50
    
    params = \
    {
        'cuda': False,
        'm': true_m,
        'K': true_K,
        'c': 0.001,     #uniformization const.
        'geo_p': 0.5,
        'inf_split': True,
        'russian_roulette': True,
        'no_russian_roulette_t': 2,
        'initialMu': 1.0,
        'modelType': 'muPerState',  # 'MMmK', 'muPerState', 'embeddedMC'
        'modelName': 'MMmK_simData1',
        'optim': 'Adam',
        'lr': 0.1,
        'use_lr_scheduler': False,
        'gradientClip': 25.0  # for exploding gradients
    }

    
    if torch.cuda.is_available():
        print 'torch.cuda.is_available() = ', torch.cuda.is_available()

    random.seed(1111)
    if params['cuda'] == True:
        torch.cuda.manual_seed(1111)
    else:
        torch.manual_seed(1111)

    model = GenericQueue_customGradients(params)
    prevMus = 'best mu= mu0=124.655592 mu1=171.366547 mu2=242.353646 mu3=347.272606 mu4=491.947895 mu5=721.457692 mu6=1062.043319 mu7=1564.547120 mu8=2363.274533 mu9=86.010762 mu10=1.000000 mu11=1.000000 mu12=1.000000 mu13=1.000000 mu14=1.000000 mu15=1.000000 mu16=1.000000 mu17=1.000000 mu18=1.000000 mu19=1.000000 mu20=1.000000 mu21=1.000000 mu22=1.000000 mu23=1.000000 mu24=1.000000 mu25=1.000000 mu26=1.000000 mu27=1.000000 mu28=1.000000 mu29=1.000000 mu30=1.000000 mu31=1.000000 mu32=1.000000 mu33=1.000000 mu34=1.000000 mu35=1.000000 mu36=1.000000 mu37=1.000000 mu38=1.000000 mu39=1.000000 mu40=1.000000 mu41=1.000000 mu42=1.000000 mu43=1.000000 mu44=1.000000 mu45=1.000000 mu46=1.000000 mu47=1.000000 mu48=1.000000 mu49=1.000000 mu50=1.000000 mu51=1.000000 mu52=1.000000 mu53=1.000000 mu54=1.000000 mu55=1.000000 mu56=1.000000 mu57=1.000000 mu58=1.000000 mu59=1.000000 mu60=1.000000 mu61=1.000000 mu62=1.000000 mu63=1.000000 mu64=1.000000 mu65=1.000000 mu66=1.000000 mu67=1.000000 mu68=1.000000 mu69=1.000000 mu70=1.000000 mu71=1.000000 mu72=1.000000 mu73=1.000000 mu74=1.000000 mu75=1.000000 mu76=1.000000 mu77=1.000000 mu78=1.000000 mu79=1.000000 mu80=1.000000 mu81=1.000000 mu82=1.000000 mu83=1.000000 mu84=1.000000 mu85=1.000483 mu86=2.185987 mu87=17.664388 mu88=95.269982 mu89=153.138020 mu90=221.555734 mu91=319.993155 mu92=466.477826 mu93=691.655807 mu94=1045.540687 mu95=1612.814318 mu96=2621.142133 mu97=4410.125033 mu98=7540.984612 '
    spltMus = prevMus.split()[2:]
    for m in range(len(spltMus)):
        parts = spltMus[m].split('=')
        model.mus[parts[0][-1]].data = torch.DoubleTensor([float(parts[-1])])

    if params['cuda'] == True:
        model.cuda()
    else:
        torch.set_num_threads(4)


    testLambdas = list(range(1225, 1400, 10))
    test_X = []
    test_Y = []
    for la in testLambdas:
        test_X = test_X + [la]*sampleCount
        PK = math.exp(solver.M_M_m_K_log(float(la) / float(true_mu), true_m, true_K))
        Y = np.random.poisson(lam = la * PK, size = sampleCount)
        test_Y = test_Y + Y.tolist()

    print 'min,max,avg,tot #drops=', min(train_Y), max(train_Y), np.mean(train_Y), sum(train_Y)
    
    
    evaluate_given_model(model, test_X, test_Y)
    
    
    

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    np.random.seed(1111)
    random.seed(1111)
    #test()
    run_using_MMmK_simulation()
    #import cProfile
    #cProfile.run('run_using_real_data()')
    #run_using_real_data()
    print('DONE!')