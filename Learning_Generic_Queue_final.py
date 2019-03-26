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
import numpy

import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


class GenericQueue(nn.Module):
    def __init__(self, params):
        super(GenericQueue, self).__init__()

        self.params = params
        self.K = params['K']
        if params['modelType'] == 'MMmK':
            self.m = params['m']
            self.mu = nn.Parameter(torch.FloatTensor([params['initialMu']]), requires_grad=True)
            

        elif params['modelType'] == 'multipleMus': # MMmK with different mu per state
            #self.mu = nn.Parameter(torch.FloatTensor([params['initialMu']] * (self.params['K'] - 1)), requires_grad=True)
            #mus = [params['initialMu']*(i+1) for i in range(self.params['K'] - 1)]
            mus = [params['initialMu'] for i in range(self.params['K'] - 1)]
            self.mu = nn.Parameter(torch.FloatTensor(mus), requires_grad=True)
            

        elif params['modelType'] == 'allMus': # each state has a different mu for all previous states
            self.mu = nn.ParameterDict()
            for i in range(1, self.params['K']):
                self.mu[str(i)] = nn.Parameter(torch.FloatTensor([params['initialMu']] * i), requires_grad=True)
                

        if params['learningTechnique'] == 'pi':
            self.pi = nn.Parameter(torch.FloatTensor([[1.0 / self.K] * self.K]), requires_grad=True)
            self.steadyState_optimiser = getattr(torch.optim, params['optim'])([self.pi], lr=params['steadyState_lr'])
            self.optimiser = getattr(torch.optim, params['optim'])([self.mu], lr=params['lr'])
            
        elif params['learningTechnique'] == 'hybrid':
            self.pi = nn.Parameter(torch.FloatTensor([[1.0 / self.K] * self.K]), requires_grad=True)
            self.optimiser = getattr(torch.optim, params['optim'])(self.parameters(), lr=params['lr'])
        else:
            self.optimiser = getattr(torch.optim, params['optim'])(self.parameters(), lr=params['lr'])
            
            

        self.Q = torch.zeros(self.params['K'], self.params['K'])
        

        paramsCount = 0
        for p in list(self.parameters()):
            if len(p.shape) == 1:
                paramsCount += p.shape[0]
            else:
                paramsCount += p.shape[0] * p.shape[1]

        print '**************************************************'
        print 'Number of parameters in the model = ', paramsCount
        print '**************************************************'
        self.paramCount = paramsCount

    def mus_2_str(self):
        str_mus = ''
        for i in range(self.paramCount):
            str_mus += 'mu%d=%f '%(i,self.mu[i].item())
        return str_mus

    def form_Q(self, inputLambda):
        if self.params['modelType'] == 'multipleMus':
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
                        self.Q[i][j] = self.mu[i - 1]
                    elif i == j:
                        if i == self.params['K'] - 1 and j == self.params['K'] - 1:
                            self.Q[i][j] = -1 * self.mu[i - 1]

                        else:
                            self.Q[i][j] = -1 * (self.mu[i - 1] + inputLambda)
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
                            self.Q[i][j] = self.mu * i
                        else:
                            self.Q[i][j] = self.mu * self.m
                    elif i == j:
                        if i == self.K - 1 and j == self.K - 1:
                            if i < self.m:
                                self.Q[i][j] = -1 * (i) * self.mu
                            else:
                                self.Q[i][j] = -1 * (self.m) * self.mu
                        else:
                            if i < self.m:
                                tmp = -1 * (self.mu * i + inputLambda)
                            else:
                                tmp = -1 * (self.mu * self.m + inputLambda)
                            self.Q[i][j] = tmp
                    else:
                        self.Q[i][j] = 0.0


        if self.params['modelType'] == 'allMus':
            if self.params['cuda'] == True:
                self.Q = torch.zeros((self.params['K'], self.params['K'])).cuda()
            else:
                self.Q = torch.zeros((self.params['K'], self.params['K']))

            self.Q[0][0] = -inputLambda
            self.Q[0][1] = inputLambda
            for i in range(1, self.params['K']):
                for j in range(0, i):
                    self.Q[i][j] = self.mu[str(i)][j]
                if i+1 < self.params['K']:
                    self.Q[i][i+1] = inputLambda
                self.Q[i][i] = -1*torch.sum(self.Q[i])

        return self.Q

    def form_const_Q(self, inputLambda):
        if self.params['modelType'] == 'multipleMus':
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
                        self.Q[i][j] = self.mu[i - 1].item()
                    elif i == j:
                        if i == self.params['K'] - 1 and j == self.params['K'] - 1:
                            self.Q[i][j] = -1 * self.mu[i - 1].item()

                        else:
                            self.Q[i][j] = -1 * (self.mu[i - 1].item() + inputLambda)
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
                            self.Q[i][j] = self.mu.item() * i
                        else:
                            self.Q[i][j] = self.mu.item() * self.m
                    elif i == j:
                        if i == self.K - 1 and j == self.K - 1:
                            if i < self.m:
                                self.Q[i][j] = -1 * (i) * self.mu.item()
                            else:
                                self.Q[i][j] = -1 * (self.m) * self.mu.item()
                        else:
                            if i < self.m:
                                tmp = -1 * (self.mu.item() * i + inputLambda)
                            else:
                                tmp = -1 * (self.mu.item() * self.m + inputLambda)
                            self.Q[i][j] = tmp
                    else:
                        self.Q[i][j] = 0.0

        elif self.params['modelType'] == 'allMus':
            if self.params['cuda'] == True:
                self.Q = torch.zeros((self.params['K'], self.params['K'])).cuda()
            else:
                self.Q = torch.zeros((self.params['K'], self.params['K']))

            self.Q[0][0] = -inputLambda
            self.Q[0][1] = inputLambda
            for i in range(1, self.params['K']):
                for j in range(0, i):
                    self.Q[i][j] = self.mu[str(i)][j].item()
                if i+1 < self.params['K']:
                    self.Q[i][i+1] = inputLambda
                self.Q[i][i] = -1*torch.sum(self.Q[i])

        return self.Q

    def form_P(self, inputLambda):
        self.form_Q(inputLambda)
        if self.params['cuda'] == True:
            I = torch.diag(torch.ones(self.params['K'])).cuda()
            g = torch.max(I * torch.abs(self.Q))
            P = torch.diag(torch.ones(self.params['K'])).cuda() + torch.div(self.Q, g).cuda()
        else:
            I = torch.diag(torch.ones(self.params['K']))
            g = torch.max(I * torch.abs(self.Q))
            P = torch.diag(torch.ones(self.params['K'])) + torch.div(self.Q, g)

        return P

    def form_const_P(self, inputLambda):
        self.form_const_Q(inputLambda)
        if self.params['cuda'] == True:
            I = torch.diag(torch.ones(self.params['K'])).cuda()
            g = torch.max(I * torch.abs(self.Q))
            P = torch.diag(torch.ones(self.params['K'])).cuda() + torch.div(self.Q, g).cuda()
        else:
            I = torch.diag(torch.ones(self.params['K']))
            g = torch.max(I * torch.abs(self.Q))
            P = torch.diag(torch.ones(self.params['K'])) + torch.div(self.Q, g)

        return P

    def zero_out_gradients_for_nonParameters(self):
        if self.Q.grad:
            for idx in self.noGradIdxs:
                self.Q.grad[idx[0]][idx[1]].fill_(0)

    def toStr(self):
        s = 'asd'
        return s

    def paramClip(self):
        if self.params['modelType'] == 'allMus':
            for i in range(1, self.params['K']):
                #self.mu[str(i)].data.clamp_(min=self.params['initialMu'], max=1e10)
                self.mu[str(i)].data.clamp_(min=1.0, max=1e10)
        else:
            self.mu.data.clamp_(min=self.params['initialMu'], max=1e10)


    def change_lr(self, lr):
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = lr

    def objective(self, inputLa, dropCount, dropProb):
        if self.params['learningTechnique'] == 'steps':  ###########################
            P = self.form_P(inputLa)
            cnt = 1
            Pt_1 = torch.mm(P, P)
            while True:
                cnt += 1
                Pt = torch.mm(Pt_1, Pt_1)
                Pt_1_K = Pt_1[-1].view(1, Pt_1.shape[0])
                if self.params['steadyStateLossType'] == 'L2':
                    steady_state_loss = F.mse_loss(torch.mm(Pt_1_K, P), Pt_1_K)
                elif self.params['steadyStateLossType'] == 'L1':
                    #steady_state_loss = torch.sum(torch.abs(torch.mm(Pt_1_K, P) - Pt_1_K))
                    steady_state_loss = torch.mean(torch.sum(torch.abs(torch.mm(Pt_1, P) - Pt_1), 1))
                    

                if cnt > self.params['t'] or steady_state_loss < self.params['steadyStateEpsilon']:
                    break

                Pt_1 = Pt

            if self.params['calculatePK'] == 'AVG':
                PK = torch.mean(Pt[:, -1])
            elif self.params['calculatePK'] == 'MIN':
                PK = torch.min(Pt[:, -1])
            elif self.params['calculatePK'] == 'MAX':
                PK = torch.max(Pt[:, -1])
            else:
                PK = torch.max(Pt[-1,-1])



        ############################################################
        elif self.params['learningTechnique'] == 'pi':
            P = self.form_const_P(inputLa)
            Pt = P
            for t in range(self.params['t']):
                Pt = torch.mm(Pt, Pt)

            self.pi.data = torch.FloatTensor([[1.0 / self.K] * self.K])
            #self.pi.data = Pt[-1].data.view(1,self.pi.data.size(1))
            cnt = 0
            #lr = self.params['steadyState_lr']
            #self.change_lr(lr)
            while True:
                cnt += 1
                pi_probs = F.softmax(self.pi, dim=1)
                pi_P = torch.mm(pi_probs, Pt)
                if self.params['steadyStateLossType'] == 'L2':
                    steady_state_loss = F.mse_loss(pi_P, pi_probs)
                elif self.params['steadyStateLossType'] == 'L1':
                    steady_state_loss = torch.sum(torch.abs(pi_P - pi_probs))

                if steady_state_loss < self.params['steadyStateEpsilon'] or cnt > self.params['steadyStateIterPerSample']:
                    break

                steady_state_loss.backward()
                # steady_state_loss.backward(retain_graph=True)
                # self.zero_out_gradients_for_nonParameters()
                # self.mu.grad.fill_(0)
                nn.utils.clip_grad_norm_(self.parameters(), self.params['gradientClip'],
                                         norm_type=self.params['gradientClipType'])
                self.steadyState_optimiser.step()
                self.steadyState_optimiser.zero_grad()
                #lr = lr/2.0
                #self.change_lr(lr)

            P = self.form_P(inputLa)

            Pt = P
            for t in range(self.params['t']):
                Pt = torch.mm(Pt, Pt)
            # logPK = F.log_softmax(torch.mm(F.softmax(self.pi, dim=1), Pt), dim=1)[-1][-1]
            PK = torch.mm(F.softmax(self.pi, dim=1), Pt)[-1][-1]
            #self.change_lr(self.params['lr'])



        else:  # hybrid ###########################################################
            P = self.form_P(inputLa)
            Pt = P
            for t in range(self.params['t']):
                Pt = torch.mm(Pt, Pt)

            self.pi.data = torch.FloatTensor([[1.0 / self.K] * self.K])
            #self.pi.data = Pt[-1].data.view(1,self.pi.data.size(1))
            cnt = 0
            #lr = self.params['steadyState_lr']
            #self.change_lr(lr)
            while True:
                cnt += 1
                pi_probs = F.softmax(self.pi, dim=1)
                pi_P = torch.mm(pi_probs, Pt)
                if self.params['steadyStateLossType'] == 'L2':
                    steady_state_loss = F.mse_loss(pi_P, pi_probs)
                elif self.params['steadyStateLossType'] == 'L1':
                    steady_state_loss = torch.sum(torch.abs(pi_P - pi_probs))

                if steady_state_loss < self.params['steadyStateEpsilon'] or cnt > self.params['steadyStateIterPerSample']:
                    break

                #steady_state_loss.backward()
                steady_state_loss.backward(retain_graph=True)
                # self.zero_out_gradients_for_nonParameters()
                # self.mu.grad.fill_(0)
                nn.utils.clip_grad_norm_(self.parameters(), self.params['gradientClip'],
                                         norm_type=self.params['gradientClipType'])
                self.optimiser.step()
                self.optimiser.zero_grad()
                #lr = lr/2.0
                #self.change_lr(lr)

            #P = self.form_P(inputLa)

            #Pt = P
            #for t in range(self.params['t']):
            #    Pt = torch.mm(Pt, Pt)
            # logPK = F.log_softmax(torch.mm(F.softmax(self.pi, dim=1), Pt), dim=1)[-1][-1]
            PK = torch.mm(F.softmax(self.pi, dim=1), Pt)[-1][-1]
            #self.change_lr(self.params['lr'])

        ############################################################

        logPK = torch.log( torch.clamp(PK, min = 1e-10, max = 1.0))

        if self.params['dataLossType'] == 'MSE':
            dataLoss = F.mse_loss(PK, torch.FloatTensor([dropProb]))
        elif self.params['dataLossType'] == 'NLL':
            dataLoss = -1 * (dropCount * logPK + (inputLa - dropCount) * torch.log(1 - torch.exp(logPK)))

        loss = self.params['dataLossWeight'] * dataLoss + self.params['steadyStateWeight'] * steady_state_loss

        return loss, logPK

    def predict(self, la, clampNumbers=False):
        if self.params['learningTechnique'] == 'steps':  ###########################
            P = self.form_P(la)
            cnt = 1
            Pt_1 = torch.mm(P, P)
            while True:
                Pt = torch.mm(Pt_1, Pt_1)
                cnt += 1
                # print Pt_1.shape, Pt_1[-1].shape, Pt_1[-1].view(1, Pt_1.shape[0]).shape
                Pt_1_K = Pt_1[-1].view(1, Pt_1.shape[0])
                if self.params['steadyStateLossType'] == 'L2':
                    steady_state_loss = F.mse_loss(torch.mm(Pt_1_K, P), Pt_1_K)
                elif self.params['steadyStateLossType'] == 'L1':
                    #steady_state_loss = torch.sum(torch.abs(torch.mm(Pt_1_K, P) - Pt_1_K))
                    steady_state_loss = torch.mean(torch.sum(torch.abs(torch.mm(Pt_1, P) - Pt_1), 1))

                if cnt > self.params['t'] or steady_state_loss < self.params['steadyStateEpsilon']:
                    break

                Pt_1 = Pt

            if self.params['calculatePK'] == 'AVG':
                PK = torch.mean(Pt[:, -1])
            elif self.params['calculatePK'] == 'MIN':
                PK = torch.min(Pt[:, -1])
            elif self.params['calculatePK'] == 'MAX':
                PK = torch.max(Pt[:, -1])
            else:
                PK = torch.max(Pt[-1,-1])

            return PK.item()

        elif self.params['learningTechnique'] == 'pi':  ###########################

            P = self.form_const_P(la)
            Pt = P
            for t in range(self.params['t']):
                Pt = torch.mm(Pt, Pt)

            #self.pi.data = torch.FloatTensor([[1.0 / self.K] * self.K])
            self.pi.data = torch.FloatTensor([[1.0 / self.K] * self.K])
            cnt = 0
            while True:
                cnt += 1
                pi_probs = F.softmax(self.pi, dim=1)
                pi_P = torch.mm(pi_probs, Pt)
                if self.params['steadyStateLossType'] == 'L2':
                    steady_state_loss = F.mse_loss(pi_P, pi_probs)
                elif self.params['steadyStateLossType'] == 'L1':
                    steady_state_loss = torch.sum(torch.abs(pi_P - pi_probs))

                if steady_state_loss < self.params['steadyStateEpsilon'] or cnt > self.params['steadyStateIterPerSample']:
                    break

                steady_state_loss.backward()
                # steady_state_loss.backward(retain_graph=True)
                # self.zero_out_gradients_for_nonParameters()
                self.mu.grad.fill_(0)
                nn.utils.clip_grad_norm_(self.parameters(), self.params['gradientClip'],
                                         norm_type=self.params['gradientClipType'])
                self.steadyState_optimiser.step()
                self.steadyState_optimiser.zero_grad()

            # P = self.form_P(inputLambda)
            # Pt = P
            # for t in range(self.params['t']):
            #    Pt = torch.mm(Pt, Pt)

            PK = torch.mm(F.softmax(self.pi, dim=1), Pt)[-1][-1]
            return PK.item()

        else:  # hybrid ###########################################################
            P = self.form_const_P(la)
            Pt = P
            for t in range(self.params['t']):
                Pt = torch.mm(Pt, Pt)

            #self.pi.data = torch.FloatTensor([[1.0 / self.K] * self.K])
            self.pi.data = torch.FloatTensor([[1.0 / self.K] * self.K])
            cnt = 0
            while True:
                cnt += 1
                pi_probs = F.softmax(self.pi, dim=1)
                pi_P = torch.mm(pi_probs, Pt)
                if self.params['steadyStateLossType'] == 'L2':
                    steady_state_loss = F.mse_loss(pi_P, pi_probs)
                elif self.params['steadyStateLossType'] == 'L1':
                    steady_state_loss = torch.sum(torch.abs(pi_P - pi_probs))

                if steady_state_loss < self.params['steadyStateEpsilon'] or cnt > self.params['steadyStateIterPerSample']:
                    break

                steady_state_loss.backward()
                # steady_state_loss.backward(retain_graph=True)
                # self.zero_out_gradients_for_nonParameters()
                self.mu.grad.fill_(0)
                nn.utils.clip_grad_norm_(self.parameters(), self.params['gradientClip'],
                                         norm_type=self.params['gradientClipType'])
                self.optimiser.step()
                self.optimiser.zero_grad()

            # P = self.form_P(inputLambda)
            # Pt = P
            # for t in range(self.params['t']):
            #    Pt = torch.mm(Pt, Pt)

            PK = torch.mm(F.softmax(self.pi, dim=1), Pt)[-1][-1]
            return PK.item()


############################################################################################################


def parseDataFile(fpath, inputPacketsCols, droppedPacketsCols):
    df = pd.read_csv(fpath, usecols=inputPacketsCols + droppedPacketsCols)
    df.fillna(0, inplace=True)  # replace missing values (NaN) to zero9
    return df


def getTrainingData(dir, summaryFile, minDropRate, maxDropRate):
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
                PK = failed / the_lambda
                # PK = failed
            except:
                continue
            train_X.append(Variable(torch.FloatTensor([the_lambda])))
            train_Y.append(Variable(torch.FloatTensor([PK])))

    return train_X, train_Y


def evaluate_given_model(testModel, testLambdas, testPKs, plotFig=True):
    testModel.eval()
    est_PKs = []
    avg_NLL = 0.0

    for i in range(len(testLambdas)):
        est_PK = testModel.predict(testLambdas[i])
        logPK = math.log(numpy.clip(est_PK, 1e-10, 1.0))
        #logPK = math.log(est_PK)
        est_PKs.append(est_PK)
        actualDrops = float(math.ceil(testPKs[i] * testLambdas[i]))
        NLL = - (actualDrops * logPK + (testLambdas[i] - actualDrops) * math.log(1 - math.exp(logPK)))
        avg_NLL += NLL

    if testModel.params['cuda'] == True:
        est_PK_torch = torch.FloatTensor(est_PKs).cuda()
        testPKs_torch = torch.FloatTensor(testPKs).cuda()
    else:
        est_PK_torch = torch.FloatTensor(est_PKs)
        testPKs_torch = torch.FloatTensor(testPKs)

    MSE = F.mse_loss(est_PK_torch, testPKs_torch)
    avg_NLL = avg_NLL / float(len(testLambdas))
    print 'Evaluation MSE=', MSE.item(), 'NLL=', avg_NLL

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
          showBatches=False
          ):
    model.train()
    bestValidLoss = 1e10
    bestModel = model

    if model.params['cuda'] == True:
        inputLambdas = Variable(torch.FloatTensor(inputLambdas)).cuda()
        PKs = Variable(torch.FloatTensor(PKs)).cuda()

        validLambdas = Variable(torch.FloatTensor(validLambdas)).cuda()
        validPKs = Variable(torch.FloatTensor(validPKs)).cuda()
    else:
        inputLambdas = Variable(torch.FloatTensor(inputLambdas))
        PKs = Variable(torch.FloatTensor(PKs))

        validLambdas = Variable(torch.FloatTensor(validLambdas))
        validPKs = Variable(torch.FloatTensor(validPKs))

    total_loss = 0.0
    for e in range(epochs):
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
        for b in range(len(inputLambdas)):

            inputLa = inputLambdas[b]
            dropProb = PKs[b]
            dropCount = float(math.ceil(dropProb * inputLa))

            loss, logPK = model.objective(inputLa, dropCount, dropProb)

            est_PKs.append(torch.exp(logPK).item())
            true_PKs.append(PKs[b])

            #print(est_PKs[-1], true_PKs[-1].item()),

            total_loss = total_loss + loss
            epochLoss += loss.item()

            if (b) % batchSize == 0:
                total_loss = total_loss / float(batchSize)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), model.params['gradientClip'],
                                         norm_type=model.params['gradientClipType'])
                model.optimiser.step()
                model.optimiser.zero_grad()
                model.paramClip()
                if showBatches:
                    print '\nbatch#', b, 'loss=', total_loss.item(), 'mu=', model.mu.data
                total_loss = 0.0

        total_loss = 0.0
        est_PKs = torch.FloatTensor(est_PKs)
        true_PKs = torch.FloatTensor(true_PKs)
        MSE = F.mse_loss(est_PKs, true_PKs)

        validMSE, validNLL = evaluate_given_model(model, validLambdas, validPKs, plotFig=False)
        #validMSE, validNLL = 0, 0
        if model.params['dataLossType'] == 'MSE':
            validLoss = validMSE
        elif model.params['dataLossType'] == 'NLL':
            validLoss = validNLL

        model.validLoss = validLoss
        model.trainLoss = epochLoss / float(len(inputLambdas))

        if validLoss < bestValidLoss:
            bestValidLoss = validLoss
            torch.save(model, model.params['modelName'])
            bestModel = torch.load(model.params['modelName'])

        print 'Epoch ',e, '--------------------------------------'
        print 'lr=', model.params['lr'],
        print 'TrainLoss= ', epochLoss / float(len(inputLambdas))
        print 'MSE=', MSE.item()
        print 'validLoss=', validLoss
        if model.params['modelType'] == 'allMus':
            mu = [model.mu[str(i)].data for i in range(1, model.params['K'])]
        else:

            mu = model.mu.data
        print 'mu=',mu
        print '\tbest validLoss=', bestModel.validLoss, 'best trainLoss=', bestModel.trainLoss
        print '\tbest mu=', [bestModel.mu[str(i)].data for i in range(1, bestModel.params['K'])]
        print

        torch.save(bestModel, bestModel.params['modelName'])


def run_using_MMmK_simulation():
    mpl.rcParams.update({'font.size': 17})
    random.seed(1111)
    true_m = 3
    true_mu = 50.0
    true_K = 5

    maxLambda = 1000
    inputDataSize = 1

    train_X = list(range(1, maxLambda, 10))
    # train_X = list(range(10, 1000, 2))
    random.shuffle(train_X)
    # train_X = train_X[:inputDataSize]

    singleValue = 600

    train_X = [singleValue]

    train_Y = [math.exp(solver.M_M_m_K_log(float(inp) / float(true_mu), true_m, true_K)) for inp in train_X]

    drops = [math.ceil(train_X[i] * train_Y[i]) for i in range(len(train_X))]

    print '#drops=', drops, train_Y
    #exit(1)
    valid_X = list(range(1, maxLambda, 7))
    # valid_X = list(range(5, 1000, 7))
    random.shuffle(valid_X)
    valid_X = [singleValue]
    #valid_X = valid_X[:inputDataSize]
    valid_Y = [math.exp(solver.M_M_m_K_log(float(inp) / float(true_mu), true_m, true_K)) for inp in valid_X]

    ########################################
    
    params = {
        'cuda': False,
        'm': true_m,
        'K': true_K,
        'initialMu': 20.0,
        'modelType': 'multipleMus',  # 'MMmK', 'multipleMus'
        'learningTechnique': 'steps',  # 'steps', 'pi', 'hybrid'
        'steadyStateIterPerSample': 1000,
        'modelName': 'genericQueueModel_MMmK_K5_m3_simulatedData_la75_pi',
        'dataLossType': 'NLL',  # NLL or  MSE
        'steadyStateLossType': 'L1',  # L1 or L2
        't': 1000,
        'steadyStateEpsilon': 1e-4,
        'dataLossWeight': 1.0,
        'steadyStateWeight': 0.0,
        'calculatePK': 'LAST',  # MIN, MAX, AVG,
        'optim': 'Adam',
        'steadyState_lr': 0.1,
        'lr': 0.1,
        'gradientClip': 10.0,
        'gradientClipType': 2
    }
    
    '''
    params = {
        'cuda': False,
        'm': 2,
        'K': 100,
        'initialMu': 25.0,
        'modelType': 'MMmK',  # 'MMmK', 'multipleMus'
        'learningTechnique': 'steps',  # 'steps', 'pi', 'hybrid'
        'steadyStateIterPerSample': 1000,
        'modelName': 'genericQueueModel_MMmK_K5_m3_simulatedData_la75',
        'dataLossType': 'NLL',  # NLL or  MSE
        'steadyStateLossType': 'L1',  # L1 or L2
        't': 1000,
        'steadyStateEpsilon': 1e-4,
        'dataLossWeight': 1.0,
        'steadyStateWeight': 0.0,
        'calculatePK': 'AVG',  # MIN, MAX, AVG,
        'optim': 'Adam',
        'steadyState_lr': 0.1,
        'lr': 0.05,
        'gradientClip': 10.0,
        'gradientClipType': 2
    }
    '''



    print
    'Parameters:\n', params

    if torch.cuda.is_available():
        print
        'torch.cuda.is_available() = ', torch.cuda.is_available()

    random.seed(1111)
    if params['cuda'] == True:
        torch.cuda.manual_seed(1111)
    else:
        torch.manual_seed(1111)

    model = GenericQueue(params)

    if params['cuda'] == True:
        model.cuda()
    else:
        torch.set_num_threads(8)

    train(model=model,
          inputLambdas=train_X, PKs=train_Y,
          batchSize=1,
          epochs=1000000,
          validLambdas=valid_X, validPKs=valid_Y,
          shuffleData=False,
          showBatches=False
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

    data_X, data_Y = getTrainingData(dir, summaryFile, minDropRate=1, maxDropRate=1e10)

    # shuffle data
    combined = list(zip(data_X, data_Y))
    random.shuffle(combined)
    data_X[:], data_Y[:] = zip(*combined)

    trainLen = int(trainQuota * len(data_X))
    validLen = int(validQuota * len(data_X))

    print 'trainLen=', trainLen, 'validLen=', validLen

    train_X = data_X[:trainLen]
    train_Y = data_Y[:trainLen]

    valid_X = data_X[trainLen:trainLen + validLen]
    valid_Y = data_Y[trainLen:trainLen + validLen]

    test_X = data_X[trainLen + validLen:]
    test_Y = data_Y[trainLen + validLen:]

    ########################################
    params = {
        'cuda': False,
        'm': 3,
        'K': 5,
        'initialMu': 500.0,
        'modelType': 'allMus',  # 'MMmK', 'multipleMus', 'allMus'
        'learningTechnique': 'steps',  # 'steps', 'pi', 'hybrid'
        'steadyStateIterPerSample': 1000,
        'modelName': 'genericQueueModel_allMus_K5_m3',
        'dataLossType': 'NLL',  # NLL or  MSE
        'steadyStateLossType': 'L1',  # L1 or L2
        't': 3,
        'steadyStateEpsilon': 1e-4,
        'dataLossWeight': 1.0,
        'steadyStateWeight': 0.0,
        'calculatePK': 'AVG',  # MIN, MAX, AVG,
        'optim': 'Adam',
        'steadyState_lr': 0.1,
        'lr': 1.0,
        'gradientClip': 10.0,
        'gradientClipType': 2
    }

    print 'Parameters:\n', params

    # print torch.__version__

    # print torch.version.cuda

    if torch.cuda.is_available():
        print
        'torch.cuda.is_available() = ', torch.cuda.is_available()

    if params['cuda'] == True:
        torch.cuda.manual_seed(1111)
    else:
        torch.manual_seed(1111)

    model = GenericQueue(params)

    if params['cuda'] == True:
        model.cuda()
    else:
        torch.set_num_threads(8)

    train(model=model,
          inputLambdas=train_X, PKs=train_Y,
          batchSize=32,
          epochs=1000,
          validLambdas=valid_X, validPKs=valid_Y,
          shuffleData=True,
          showBatches=False
          )


def test():
    true_m = 3
    true_mu = 50.0
    true_K = 5
    params = {
        'cuda': False,
        'K': 5,
        'm': 3,
        'initialMu': 50.0,
        'modelName': 'mimicing_MMmK',
        'dataLossType': 'NLL',  # NLL or  MSE
        'steadyStateLossType': 'L1',  # L1 or L2
        't': 10,
        'steadyStateEpsilon': 10e-8,
        'dataLossWeight': 1.0,
        'steadyStateWeight': 0.0,
        'calculatePK': 'MIN',  # MIN, MAX, AVG,
        'optim': 'Adam',
        'lr': 0.1,
        'gradientClip': 10.0,
        'gradientClipType': 2
    }
    model = GenericQueue(params)
    # testLambdas = list(range(15,1000,10))
    testLambdas = [500]
    # testPKs = [ solver.M_M_m_K_getProbAt_k(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in testLambdas ]
    testPKs = [math.exp(solver.M_M_m_K_log(float(inp) / float(true_mu), true_m, true_K)) for inp in testLambdas]

    evaluate_given_model(model, testLambdas, testPKs)


if __name__ == "__main__":
    # test()
    #run_using_MMmK_simulation()
    run_using_real_data()
    print('DONE!')