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

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt



class learn_genericQueue_fromData(nn.Module):
    def __init__(self, initialMu = 10, K=10, lossType = 'MSE', power_P = 5, eps = 0.01,
                 lr=0.01, gradientClip=10.0, gradientClipType='inf', optim='SGD'):
        super(learn_genericQueue_fromData, self).__init__()
       
       
        self.K = K
        self.power_P = power_P
        self.lossType = lossType
        self.eps = eps
        
        self.mu = nn.Parameter(torch.FloatTensor([initialMu]*(K-1)),requires_grad = True)
        #self.mu = nn.Parameter(torch.FloatTensor([10,20,30,40]),requires_grad = True)
        
        self.Q = torch.zeros(K,K)
       
        self.dataLossWeight = 1.0
        self.steadyState_weight  = 0.0

        self.lr = lr
        self.gradientClip = gradientClip
        self.gradientClipType = gradientClipType
        self.optim = optim
        self.optimiser = getattr(torch.optim, self.optim)(self.parameters(), lr=self.lr)
        # self.optimiser = torch.optim.SGD(self.parameters(), self.lr)
        paramsCount = 0
        for p in list(self.parameters()):
            if len(p.shape) == 1:
                paramsCount += p.shape[0]
            else:
                paramsCount += p.shape[0] * p.shape[1]

        print '**************************************************'
        print 'Number of parameters in the model = ', paramsCount
        print '**************************************************'
        
    
    def form_Q(self, inputLambda):
        #self.Q = nn.Parameter(torch.zeros(self.K,self.K),requires_grad = True)
        self.Q = torch.zeros((self.K, self.K))
        #self.noGradIdxs = [] 
        for i in range(self.K):
            for j in range(self.K):
                if i == 0 and j == 0:
                    self.Q[i][j] = -inputLambda
                    #self.noGradIdxs.append([i,j])
                elif i == j-1:
                    self.Q[i][j] = inputLambda
                    #self.noGradIdxs.append([i,j])
                elif i-1 == j:
                    self.Q[i][j] = self.mu[i-1]
                elif i == j:
                    if i == self.K - 1 and j == self.K - 1:
                        self.Q[i][j] = -1*self.mu[i-1]
                            
                    else:
                        self.Q[i][j] = -1*(self.mu[i-1] + inputLambda)
                else:
                    self.Q[i][j] = 0.0
                    #self.noGradIdxs.append([i,j])
                
    def zero_out_gradients_for_nonParameters(self):
        if self.Q.grad:
            for idx in self.noGradIdxs:
                self.Q.grad[idx[0]][idx[1]].fill_(0)   
                
    def toStr(self):
        s = 'asd'
        return s

    def paramClip(self):
        self.mu.data.clamp_(min=1.0, max=1e10)

    def forwardPass(self, inp):
        pass
            #print '\t', steady_state_loss.item()
            #print '\t', self.mu.item()

        
        #return probs[-1][-1]
        


    def objective(self, inputLa, dropCount, dropProb):
        self.form_Q(inputLa)
        g = torch.max ( torch.diag(torch.ones(self.K)) * torch.abs(self.Q) )
        P = torch.diag(torch.ones(self.K)) + torch.div(self.Q, g)
        cnt = 1
        while True:
            P = torch.mm(P,P)
            #print P
            cnt += 1
            pi = torch.FloatTensor([P.data[0].numpy().tolist()])
            #dist = F.mse_loss(torch.mm(pi,P), pi)
            dist = torch.mean(torch.abs(torch.mm(pi,P) - pi))
            if cnt > self.power_P or dist < self.eps:
            #if cnt > self.power_P:
                break
        
        '''    
        self.pi.data = torch.FloatTensor([P.data[0].numpy().tolist()])
        for i in range(self.iterPerSample):
            
            pi_probs = F.softmax(self.pi, dim=1)
            pi_P = torch.mm(pi_probs, P)
            steady_state_loss = F.mse_loss( pi_P, pi_probs)
            #print '\tsteadyStateLoss=',steady_state_loss.item()
            
            steady_state_loss.backward(retain_graph=True)
            #steady_state_loss.backward(retain_graph=True)
            self.zero_out_gradients_for_nonParameters()
            self.mu.grad.fill_(0)
            nn.utils.clip_grad_norm_(self.parameters(), self.gradientClip, norm_type=self.gradientClipType)
            self.optimiser.step()
            self.optimiser.zero_grad()
        '''
        
        #self.form_Q(inputLa)
        #g = torch.max ( torch.diag(torch.ones(self.K)) * torch.abs(self.Q) )
        #P = torch.diag(torch.ones(self.K)) + torch.div(self.Q, g)
        
        PK = P[-1][-1]
        logPK = torch.log(PK)
        #logPK = F.log_softmax(torch.mm(F.softmax(self.pi, dim=1), P), dim = 1)[-1][-1]
        
        # NLL = - [ (#dropped calls)*logPK + (#not dropped packets)*(1-logPK) ]
        #NLL = -torch.sum(dropCount * logPK + (inputLa - dropCount) * torch.log(1 - torch.exp(logPK)))
        
        if self.lossType == 'MSE':
            dataLoss = F.mse_loss(PK, torch.FloatTensor([dropProb]))
        elif self.lossType == 'NLL':
            dataLoss = -torch.sum(dropCount * logPK + (inputLa - dropCount) * torch.log(1 - torch.exp(logPK)))
        
        #steady_state_loss = F.mse_loss( torch.mm(F.softmax(self.pi, dim=1), P), F.softmax(self.pi, dim=1))
        steady_state_loss = 0.0
        
        loss = self.dataLossWeight * dataLoss + self.steadyState_weight * steady_state_loss
        

        return loss, logPK
    
    def predict(self, la, clampNumbers = False):
        self.form_Q(la)
        g = torch.max(torch.diag(torch.ones(self.K)) * torch.abs(self.Q))
        P = torch.diag(torch.ones(self.K)) + torch.div(self.Q, g)
        cnt = 1
        while True:
            P = torch.mm(P, P)
            cnt += 1
            pi = torch.FloatTensor([P.data[0].numpy().tolist()])
            dist = F.mse_loss(torch.mm(pi, P), pi)
            if cnt > self.power_P or dist < self.eps:
                break
        PK = P[-1][-1]
        return PK.item()

############################################################################################################


def parseDataFile(fpath, inputPacketsCols, droppedPacketsCols):
    df = pd.read_csv(fpath, usecols = inputPacketsCols+droppedPacketsCols)
    df.fillna(0, inplace=True) # replace missing values (NaN) to zero9
    return df


def getTrainingData(dir, summaryFile, minDropRate, maxDropRate):
    sfile = dir+summaryFile
    inputPacketsCols = ['CallRate(P)']
    droppedPacketsCols = ['FailedCall(P)']
    
    df = pd.read_csv(sfile, usecols = ['Rate File', ' Failed Calls'])
    df.fillna(0, inplace=True)
    train_X = []
    train_Y = [] 
    for i, row in df.iterrows():
        if row[' Failed Calls'] < minDropRate or row[' Failed Calls'] > maxDropRate:
            continue
        fname = 'sipp_data_' + row['Rate File'] + '_1.csv'
        simulationFile = dir + fname #sipp_data_UFF_Perdue_01_1_reduced_1.csv     UFF_Perdue_01_12_reduced
        
        curr_df = pd.read_csv(simulationFile, usecols = inputPacketsCols+droppedPacketsCols)
        curr_df.fillna(0, inplace=True) # replace missing values (NaN) to zero9
        for j, curr_row in curr_df.iterrows():
            try:
                the_lambda = float(curr_row['CallRate(P)'])
                failed = float(curr_row['FailedCall(P)'])
                if failed > the_lambda or the_lambda <= 0:
                    continue
                #PK = failed/the_lambda
                PK = failed
            except:
                continue
            train_X.append(Variable(torch.FloatTensor([the_lambda])))
            train_Y.append(Variable(torch.FloatTensor([PK])))
            
    return train_X, train_Y
 



def predict_PK(model, la):
    model.form_Q(la)
    g = torch.max ( torch.diag(torch.ones(model.K)) * torch.abs(model.Q) )
    P = torch.diag(torch.ones(model.K)) + torch.div(model.Q, g)
    cnt = 1
    while True:
        P = torch.mm(P,P)
        cnt += 1
        pi = torch.FloatTensor([P.data[0].numpy().tolist()])
        dist = F.mse_loss(torch.mm(pi,P), pi)
        if cnt > model.power_P or dist < model.eps:
            break
    PK = P[-1][-1]
    return PK


    
def evaluate_given_model(testModel, testLambdas, testPKs, plotFig = True):
    est_PKs = []
    avg_NLL = 0.0
    for i in range(len(testLambdas)):
        est_PK = predict_PK(testModel, testLambdas[i])
        logPK = torch.log(est_PK)
        est_PKs.append(est_PK.item())
        actualDrops = testPKs[i] * testLambdas[i]
        NLL = -torch.sum(actualDrops * logPK + (testLambdas[i] - actualDrops) * torch.log(1 - torch.exp(logPK)))
        avg_NLL += NLL.item()
        
    est_PK_torch = torch.FloatTensor(est_PKs)
    true_PKs_torch = torch.FloatTensor(testPKs)
    MSE = F.mse_loss(est_PK_torch, true_PKs_torch)
    avg_NLL = avg_NLL / float(len(testLambdas))
    print 'Evaluation MSE=', MSE.item(), 'NLL=', avg_NLL
        
    if plotFig:  
        fig = plt.figure(1, figsize=(6, 4))
        axes = plt.gca()
        ax = plt.axes()
        
        #drawing est PK vs. emp PK
        plt.ylabel('Probability')
        plt.xlabel('Lambda')
        lines = plt.plot(testLambdas, est_PKs, '--r' ,label='Estimated PK') 
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(testLambdas, testPKs, 'b' ,label='True PK') 
        plt.setp(lines, linewidth=2.0)
        plt.legend(loc = 2, prop={'size':17}, labelspacing=0.1) 
        fig.suptitle('Estimated PK Vs. True PK', fontsize=12, fontweight='bold', horizontalalignment='center', y=.86)
        plt.grid()                                                                     
        plt.show()
        
    return MSE.item(), avg_NLL
    
        


def train(model = None, 
          inputLambdas = [], PKs = [], 
          batchSize=10, 
          epochs = 150,
          validLambdas = [], validPKs = [],
          modelName = 'abc',
          shuffleData = True,
          showBatches = False
          ):
    
    model.train()
    bestValidLoss = 1e100
    bestModel = model
    
    inputLambdas = Variable(torch.FloatTensor(inputLambdas))
    PKs = Variable(torch.FloatTensor(PKs))
    
    validLambdas = Variable(torch.FloatTensor(validLambdas))
    validPKs = Variable(torch.FloatTensor(validPKs))
     
    total_loss = 0.0
    for e in range(epochs):
        #shuffle training data
        if shuffleData:
            rows = len(inputLambdas)
            idxs = list(range(0,rows))
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
            dropCount = dropProb * inputLa
            
            loss, logPK = model.objective(inputLa, dropCount, dropProb)
            
            est_PKs.append(torch.exp(logPK).item())
            true_PKs.append(PKs[b])
            
            total_loss = total_loss + loss
            epochLoss += loss.item()
            
            if b % batchSize == 0 and b != 0:
                total_loss = total_loss / float(batchSize) 
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), model.gradientClip, norm_type=model.gradientClipType)
                model.optimiser.step()
                model.optimiser.zero_grad()
                if showBatches:
                    print 'batch#',b ,'loss=',total_loss.item()
                total_loss = 0.0
                
                
        est_PKs = torch.FloatTensor(est_PKs)
        true_PKs = torch.FloatTensor(true_PKs)
        MSE = F.mse_loss(est_PKs, true_PKs)
                
        validMSE, validNLL = evaluate_given_model(model, validLambdas, validPKs, plotFig = False)
        if model.lossType == 'MSE':
            validLoss = validMSE
        elif model.lossType == 'NLL':
            validLoss = validNLL
        
        if validLoss < bestValidLoss:
            bestValidLoss = validLoss
            torch.save(model, modelName)
            bestModel = torch.load(modelName)
            
        print 'epoch=', e , 'lr=', model.lr, 'mu=',model.mu.data
        print 'TrainLoss= ', epochLoss/float(len(inputLambdas))
        print 'MSE=', MSE.item()
        print 'validLoss=', validLoss
        print '\tbest validLoss=', bestValidLoss
        #print '\tbest mu=',bestModel.mu.data
        print
        
        torch.save(bestModel, modelName)
    
   
   
   



def main():
    random.seed(1111)
    torch.manual_seed(1111)
    
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_4_K_DEFT_SCALE_86_2018.10.29-22.19.41/sipp_results/'
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_2_K_100000_SCALE_43_2018.11.03-13.38.21/sipp_results/'
    
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.4+.5_CORE_1_K_425982_SCALE_24_2018.11.23-18.47.32/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_INVITE_CORE_1_K_425982_SCALE_17_2018.11.21-02.49.02/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_MSG_CORE_1_K_425982_SCALE_80_2018.11.21-04.56.39/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_SUBSCRIBE_CORE_1_K_425982_SCALE_36_2018.11.21-07.04.10/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.5+0_CORE_1_K_425982_SCALE_30_2018.11.23-20.55.11/sipp_results/'
    
    summaryFile = 'summary_data_dump.csv'
    
    trainQuota = 0.85
    validQuota = 0.15
    
    data_X, data_Y = getTrainingData(dir, summaryFile, minDropRate=1, maxDropRate=10)
    
    #shuffle data
    combined = list(zip(data_X, data_Y))
    random.shuffle(combined)
    data_X[:], data_Y[:] = zip(*combined)
    
    trainLen = int(trainQuota*len(data_X))
    validLen = int(validQuota*len(data_X))
    
    print 'trainLen=', trainLen, 'validLen=', validLen
    
    train_X = data_X[:trainLen]
    train_Y = data_Y[:trainLen]
    
    valid_X = data_X[trainLen:trainLen+validLen]
    valid_Y = data_Y[trainLen:trainLen+validLen]
    
    test_X = data_X[trainLen+validLen:]
    test_Y = data_Y[trainLen+validLen:]
    
    ########################################
    K = 10
    modelName = 'genericQueue_fromData_'+str(K)
    model = learn_genericQueue_fromData(
        initialMu = 200.0,
        K = K, 
        lossType = 'NLL', #'MSE', 'NLL'
        power_P = min(K, 10),
        eps = 0.00001,
        lr=0.05, gradientClip=5.0, gradientClipType=2, optim='Adam'
        )
    
    
    
    train(model = model, 
          inputLambdas = train_X, PKs = train_Y, 
          batchSize = 32, 
          epochs = 100,
          validLambdas = valid_X, validPKs = valid_Y,
          modelName = modelName,
          shuffleData = True,
          showBatches = True
          )
    
    
  

def test():
    true_m = 3
    true_mu = 50.0
    true_K = 5
    model = learn_genericQueue_fromData(
        initialMu = true_mu, 
        K = true_K, 
        lossType = 'NLL', #'MSE', 'NLL'
        power_P = min(true_K, 10),#min(true_K, 10),
        eps = 1e-20,
        lr=0.01, gradientClip=0.25, gradientClipType=2, optim='Adam'
        )
    testLambdas = list(range(15,1000,10))
    #testPKs = [ solver.M_M_m_K_getProbAt_k(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in testLambdas ]
    testPKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K) )  for inp in testLambdas ]
    
    evaluate_given_model(model, testLambdas, testPKs)

if __name__ == "__main__":
    #test()
    main()
    print('DONE!')