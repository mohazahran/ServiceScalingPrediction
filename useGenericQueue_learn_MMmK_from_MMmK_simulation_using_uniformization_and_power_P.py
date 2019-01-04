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



class useGenericQueue_learn_MMmK(nn.Module):
    def __init__(self, initialm = 2, initialMu = 10, K=10, lossType = 'MSE', power_P = 5, eps = 0.01,
                 lr=0.01, gradientClip=10.0, gradientClipType='inf', optim='SGD'):
        super(useGenericQueue_learn_MMmK, self).__init__()
       
       
        self.K = K
        self.power_P = power_P
        self.lossType = lossType
        self.eps = eps
        
        self.mu = nn.Parameter(torch.FloatTensor([initialMu]),requires_grad = True)
        self.m = initialm
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
                    self.Q[i][j].data.clamp_(min = -inputLambda, max = -inputLambda)
                    #self.noGradIdxs.append([i,j])
                elif i == j-1:
                    self.Q[i][j].data.clamp_(min = inputLambda, max = inputLambda)
                    #self.noGradIdxs.append([i,j])
                elif i-1 == j:
                    #self.Q.weight[i][j].data.clamp_(min = self.initial_mu, max = self.initial_mu)
                    if i < self.m:
                        self.Q[i][j] = self.mu * i
                    else:
                        self.Q[i][j] = self.mu * self.m
                elif i == j:
                    if i == self.K - 1 and j == self.K - 1:
                        #self.Q.weight[i][j].data.clamp_(min = -self.initial_mu, max = -self.initial_mu)
                        if i < self.m:
                            #self.Q[i][j] = -1*(i-2)*self.mu
                            self.Q[i][j] = -1*(i)*self.mu
                        else:
                            self.Q[i][j] = -1*(self.m)*self.mu
                            
                    else:
                        #tmp = -1*(self.initial_mu + inputLambda)
                        #self.Q.weight[i][j].data.clamp_(min=tmp, max =tmp)
                        if i < self.m:
                            tmp = -1*(self.mu*i + inputLambda)
                        else:
                            tmp = -1*(self.mu*self.m + inputLambda)
                        self.Q[i][j] = tmp
                else:
                    self.Q[i][j].data.clamp_(min=0.0, max = 0.0)
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
            dist = F.mse_loss(torch.mm(pi,P), pi)
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
    
 

############################################################################################################





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
          modelName = 'abc'):
    
    model.train()
    bestValidLoss = 1e100
    bestModel = model
     
    total_loss = 0.0
    for e in range(epochs):
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
            
            if b % batchSize == 0:
                total_loss = total_loss / float(batchSize) 
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), model.gradientClip, norm_type=model.gradientClipType)
                model.optimiser.step()
                model.optimiser.zero_grad()
                #print 'batch loss=', total_loss.item(), 'mu=', model.mu.item()
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
            
        print 'epoch=', e , 'lr=', model.lr, 'mu=',model.mu.item()
        print 'TrainLoss= ', epochLoss/float(len(inputLambdas))
        print 'MSE=', MSE.item()
        print 'validLoss=', validLoss
        print '\tbest validLoss=', bestValidLoss
        print '\tbest mu=',bestModel.mu.item()
        print
        
        torch.save(bestModel, modelName)
    
   
   
   



def main():
    random.seed(1111)
    torch.manual_seed(1111)
    inputLambdas = list(range(10,1000,2))
    random.shuffle(inputLambdas)
    
    true_m = 3
    true_mu = 30.0
    true_K = 10
    modelName = 'abc'
    
    #PKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K)) for inp in inputLambdas ]
    #PKs = [ solver.M_M_m_K_getProbAt_k_2(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in inputLambdas ]
    #PKs = [ solver.M_M_m_K_getProbAt_k(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in inputLambdas ]
    PKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K) )  for inp in inputLambdas ]
    
    model = useGenericQueue_learn_MMmK(
        initialm = 1,
        initialMu = 75.0,
        K = 20,
        lossType = 'NLL', #'MSE', 'NLL'
        power_P = min(true_K, 10),
        eps = 0.00001,
        lr=0.05, gradientClip=5.0, gradientClipType=2, optim='Adam'
        )
    

    validLambdas = list(range(5,1000,7))
    #validPKs = [ solver.M_M_m_K_getProbAt_k(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in validLambdas ]
    validPKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K) )  for inp in validLambdas ]
    
    train(model = model, 
          inputLambdas = inputLambdas, PKs = PKs, 
          batchSize = 10, 
          epochs = 50,
          validLambdas = validLambdas, validPKs = validPKs,
          modelName = modelName)
    
    
    
    testLambdas = list(range(15,1000,10))
    #testPKs = [ solver.M_M_m_K_getProbAt_k(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in testLambdas ]
    testPKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K) )  for inp in testLambdas ]
    
    bestModel = torch.load(modelName)
    evaluate_given_model(bestModel, testLambdas, testPKs)
    

def test():
    true_m = 10
    true_mu = 25.0
    true_K = 150
    model = useGenericQueue_learn_MMmK(
        initialm = true_m, 
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