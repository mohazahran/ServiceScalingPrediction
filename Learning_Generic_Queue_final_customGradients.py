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

mpl.use('TkAgg')
import matplotlib.pyplot as plt


class GenericQueue_customGradients(nn.Module):
    def __init__(self, params):
        super(GenericQueue_customGradients, self).__init__()

        self.params = params
        self.K = params['K']
        self.c = params['c']
        if params['modelType'] == 'MMmK':
            self.m = float(params['m'])
            self.mu = nn.Parameter(torch.DoubleTensor([params['initialMu']]), requires_grad=True)
            #self.mu = torch.DoubleTensor([params['initialMu']])
            #self.mu = float(params['initialMu'])

        self.optimiser = getattr(torch.optim, params['optim'])(self.parameters(), lr=params['lr'])
        if self.params['use_lr_scheduler']:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=[10,20,30,100,200,300,400,500], gamma=0.5)


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
        dQ = torch.zeros((self.K, self.K))
        for i in range(1, self.K):
            if i < self.m:
                dQ[i,i-1] = i
                dQ[i,i] = -i
            else:
                dQ[i,i-1] = self.m
                dQ[i,i] = -self.m
        return dQ
      
    def form_dP_dmu(self, P):
        dP_dmu = torch.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                try:
                    P[i][j].backward(retain_graph = True)
                    dP_dmu[i][j] = self.mu.grad
                    self.mu.grad.zero_()
                except:
                    pass
        return dP_dmu
              
    def E_geometric(self, geo_p, P, dP_dmu):
        '''
        #x = np.random.geometric(geo_p)
        x = scipy.stats.geom.rvs(geo_p)
        Pr_x = scipy.stats.geom.pmf(x, geo_p)
        P_inv = np.linalg.inv(P)
        P_power_i_1 = np.eye(self.K, self.K)
        
        expectn = np.zeros((self.K, self.K))
        for i in range(1, x+1):
            ex = P_inv.dot(dP_dmu)
            ex = ex.dot(P_power_i_1)
            one_cdf_i = scipy.stats.geom.sf(i, geo_p)
            ex = ex / one_cdf_i
            expectn = expectn + ex
            
            P_power_i_1 = P_power_i_1.dot(P)
        
        expectn = expectn * Pr_x
        return expectn
        '''
        #x = 100
        #while not x<2:
        #    x = scipy.stats.geom.rvs(geo_p)
        x = scipy.stats.geom.rvs(geo_p)
        Pr_x = scipy.stats.geom.pmf(x, geo_p)
        P_inv = torch.inverse(P)
        P_inv_power_neg_i = P_inv
        P_power_i_1 = torch.diag(torch.ones(self.params['K']))

        expectn = torch.zeros((self.K, self.K))
        for i in range(1, x + 1):
            ex = torch.mm(P_inv_power_neg_i, dP_dmu)
            ex = torch.mm(ex, P_power_i_1)
            one_cdf_i = scipy.stats.geom.sf(i, geo_p)
            ex = torch.div(ex, one_cdf_i)
            expectn = expectn + ex
            if i+1 < x+1:
                P_power_i_1 = torch.mm(P_power_i_1, P)
                P_inv_power_neg_i = torch.mm(P_inv_power_neg_i, P_inv)

        expectn = torch.mul(expectn, Pr_x)
        return expectn
            
            
            
        
            
            

    def objective(self, inputLa, dropCount, dropProb):
        '''
        Q = self.form_Q(inputLa)
        I = np.eye(self.K, self.K)
        gamma = np.max(I * np.abs(self.Q)) + self.c
        P = np.eye(self.K, self.K) + Q / gamma

        Pprime = P - torch.DoubleTensor(np.eye(self.K, self.K))
        Pprime = np.concatenate((Pprime,np.ones((self.K,1))), axis = 1)
        b = np.zeros((1, self.K+1))
        b[-1][-1] = 1
        pi, residues, rank, s = np.linalg.lstsq(Pprime.T, b.T)
        
        PK = pi[-1][-1]  #g(H)
        
        dNLL = (-dropCount/PK) + (inputLa-dropCount)/(1-PK)
        
        dg_dH = np.zeros((self.K, self.K))
        dg_dH[-1][-1] = 1.0
        
        dQ_dmu = self.form_dQ_dmu()
        
        dgamma_dmu = self.m
        
        #dP_dQ = np.ones((self.K, self.K)) * gamma
        #E = self.E_geometric(0.5, P, dP_dQ)
        
        dP_dmu = ((gamma * dQ_dmu) - (Q * dgamma_dmu)) / (gamma ** 2)
        
        E = self.E_geometric(0.5, P, dP_dmu)
        
        dLoss_dmu = dNLL * dg_dH * E
        
        dLoss_dmu = dLoss_dmu[-1][-1]
        
        self.mu = self.mu - self.params['lr'] * dLoss_dmu
        

        loss = (-1 * dropCount * np.log(PK)) - ((inputLa - dropCount) * np.log(1 - PK))
        
        return loss, np.log(PK)
        '''
        
        Q = self.form_Q(inputLa)
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
        
        PK = torch.DoubleTensor([pi[-1][-1]])  #g(H)
        PK = np.clip(PK, 1e-10, 1.0)
        dNLL = (-dropCount/PK) + (inputLa-dropCount)/(1-PK)
        dNLL = dNLL.double()
        
        dg_dH = torch.zeros((self.K, self.K))
        dg_dH[-1][-1] = 1.0
        #dg_dH[:,self.K-1] = 1.0/self.K
        
        self.form_dP_dmu(P)
        
        #dQ_dmu = self.form_dQ_dmu()
        #dgamma_dmu = self.m
        
        #dP_dmu = torch.div( (torch.mul(dQ_dmu, gamma) - torch.mul(Q, dgamma_dmu)) , (gamma ** 2) )
        dP_dmu = form_dP_dmu(P)
        #dP_dmu = ((gamma * dQ_dmu) - (Q * dgamma_dmu)) / (gamma ** 2)
        
        #E = self.E_geometric(self.params['geo_p'], P, dP_dmu)
         
        #dLoss_dmu = dNLL.item() * torch.sum(dg_dH * E).item()
        
        iters = 1
        grads = []
        for j in range(iters):
            E = self.E_geometric(self.params['geo_p'], P, dP_dmu)
            dLoss_dmu = dNLL.item() * torch.sum(dg_dH * E).item()
            grads.append(dLoss_dmu)
        avgGrad = sum(grads)/float(len(grads))
        varGrad = sum([(g-avgGrad)**2 for g in grads])/float(len(grads))
        #print 'mu=%f Geo_p=%f avgGrad=%f var=%f' % (self.mu.item(), self.params['geo_p'], avgGrad, varGrad) 
            
        
        
        
        try:
            loss = (-1 * dropCount * math.log(PK)) - ((inputLa - dropCount) * math.log(1 - PK))
        except:
            print PK
            exit(1)
        
        return loss, torch.log(PK), avgGrad
        
        
        




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
            train_X.append(Variable(torch.DoubleTensor([the_lambda])))
            train_Y.append(Variable(torch.DoubleTensor([PK])))

    return train_X, train_Y


def evaluate_given_model(testModel, testLambdas, testPKs, plotFig=True):
    testModel.eval()
    est_PKs = []
    avg_NLL = 0.0

    for i in range(len(testLambdas)):
        est_PK = testModel.predict(testLambdas[i])
        logPK = math.log(np.clip(est_PK, 1e-10, 1.0))
        # logPK = math.log(est_PK)
        est_PKs.append(est_PK)
        actualDrops = float(math.ceil(testPKs[i] * testLambdas[i]))
        NLL = - (actualDrops * logPK + (testLambdas[i] - actualDrops) * math.log(1 - math.exp(logPK)))
        avg_NLL += NLL

    if testModel.params['cuda'] == True:
        est_PK_torch = torch.DoubleTensor(est_PKs).cuda()
        testPKs_torch = torch.DoubleTensor(testPKs).cuda()
    else:
        est_PK_torch = torch.DoubleTensor(est_PKs)
        testPKs_torch = torch.DoubleTensor(testPKs)

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
        if e == 63:
            dbg = 1
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
            dropProb = PKs[b]
            dropCount = float(math.ceil(dropProb * inputLa))

            loss, logPK, dLoss_dmu = model.objective(inputLa, dropCount, dropProb)
            
            gradient += dLoss_dmu

            est_PKs.append(torch.exp(logPK).item())
            true_PKs.append(PKs[b])

            #print(est_PKs[-1], true_PKs[-1].item()),

            total_loss = total_loss + loss
            epochLoss += loss.item()

            if (b) % batchSize == 0:
                total_loss = total_loss / float(batchSize)
                gradient /= float(batchSize)
                gradient = np.clip(gradient, -1*model.params['gradientClip'], model.params['gradientClip'])
                model.mu.grad = torch.DoubleTensor([gradient])
                #model.mu.data = model.mu.data - model.params['lr'] * gradient
                #model.mu.data.clamp_(min = 1.0, max = 1e10)
                #print 'new mu=', model.mu.item()
                #print model.mu, gradient
                model.optimiser.step()
                model.mu.data.clamp_(min = 1.0, max = 1e10)
                for group in model.optimiser.param_groups:
                    for p in group['params']:
                       if p.grad is not None:
                           p.grad.detach()
                           p.grad.zero_()
                
                #model.optimiser.zero_grad()
                #model.paramClip()

                if showBatches:
                    print 'batch#', b, 'loss=', total_loss.item(), 'mu=', model.mu.item(), 'grad=', gradient
                gradient = 0.0
                total_loss = 0.0

        total_loss = 0.0
        est_PKs = torch.DoubleTensor(est_PKs)
        true_PKs = torch.DoubleTensor(true_PKs)
        MSE = F.mse_loss(est_PKs, true_PKs)

        validMSE, validNLL = evaluate_given_model(model, validLambdas, validPKs, plotFig=False)
        #validMSE, validNLL = 0, 0
        #if model.params['dataLossType'] == 'MSE':
        #    validLoss = validMSE
        #elif model.params['dataLossType'] == 'NLL':
        #    validLoss = validNLL
        validLoss = validNLL

        model.validLoss = validLoss
        model.trainLoss = epochLoss / float(len(inputLambdas))

        
        
        print 'Epoch ',e, '--------------------------------------'
        for param_group in model.optimiser.param_groups:
            print 'lr=', param_group['lr'],
        print 'TrainLoss= ', epochLoss / float(len(inputLambdas))
        print 'MSE=', MSE.item()
        print 'validLoss=', validLoss
        if model.params['modelType'] == 'allMus':
            mu = [model.mu[str(i)].data for i in range(1, model.params['K'])]
        else:

            mu = model.mu.item()
        print 'mu=',mu
        
        
        if validLoss < bestValidLoss:
            bestValidLoss = validLoss
            torch.save(model, model.params['modelName'])
            bestModel = torch.load(model.params['modelName'])
        else:
            pass
            #model.params['lr'] /= 2.0

        print '\tbest validLoss=', bestModel.validLoss.item(), 'best trainLoss=', bestModel.trainLoss
        print '\tbest mu=',bestModel.mu.item()
        print

        torch.save(bestModel, bestModel.params['modelName'])
        
            
        


def run_using_MMmK_simulation():
    mpl.rcParams.update({'font.size': 17})
    random.seed(1111)
    true_m = 3
    true_K = 5
    true_mu = 50.0
    
    
    singleValue = 600

    maxLambda = 1000
    inputDataSize = 1
    

    train_X = list(range(1, maxLambda, 10))
    # train_X = list(range(10, 1000, 2))
    random.shuffle(train_X)
    # train_X = train_X[:inputDataSize]
    #singleValue = 200
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
        'c': 0.001,           #uniformization const.
        'geo_p': 0.5,
        'initialMu': 25.0,
        'modelType': 'MMmK',  # 'MMmK', 'multipleMus'
        'modelName': 'genericQueueModel_MMmK_K5_m3_simulatedData_customGradient',
        'optim': 'SGD',
        'lr': 0.1,
        'use_lr_scheduler': True,
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
        torch.set_num_threads(8)

    train(model=model,
          inputLambdas=train_X, PKs=train_Y,
          batchSize=1,
          epochs=500,
          validLambdas=valid_X, validPKs=valid_Y,
          shuffleData=False,
          showBatches=True
          )

    testLambdas = list(range(1, maxLambda, 1))
    # testLambdas = list(range(15, 1000, 10))
    testPKs = [math.exp(solver.M_M_m_K_log(float(inp) / float(true_mu), true_m, true_K)) for inp in testLambdas]
    evaluate_given_model(model, testLambdas, testPKs)


def run_using_real_data():
    random.seed(1111)

    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_INVITE_CORE_1_K_425982_SCALE_60_REMOTE_CPU_2019.01.08-01.56.08/sipp_results/'
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

    data_X, data_Y = getTrainingData(dir, summaryFile, minDropRate=1, maxDropRate=10)

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
        'c': 0.001,           #uniformization const.
        'geo_p': 0.95,
        'initialMu': 900.0,
        'modelType': 'MMmK',  # 'MMmK', 'multipleMus'
        'modelName': 'genericQueueModel_MMmK_K5_m3_realData_customGradient',
        'optim': 'Adam',
        'lr': 0.1,
        'use_lr_scheduler': False,
        'gradientClip': 5.0 # for exploding gradients
    }

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

    if params['cuda'] == True:
        model.cuda()
    else:
        torch.set_num_threads(8)

    train(model=model,
          inputLambdas=train_X, PKs=train_Y,
          batchSize=32,
          epochs=500,
          validLambdas=valid_X, validPKs=valid_Y,
          shuffleData=True,
          showBatches=True
          )


if __name__ == "__main__":
    np.random.seed(1111)
    random.seed(1111)
    # test()
    run_using_MMmK_simulation()
    #run_using_real_data()
    print('DONE!')