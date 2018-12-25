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
    def __init__(self, initialm = 2, initialMu = 10, K=10, lr=0.01, gradientClip=10.0, gradientClipType='inf', optim='SGD'):
        super(useGenericQueue_learn_MMmK, self).__init__()
       
       
        self.K = K
        self.initial_mu = initialMu
        self.initial_m = initialm
        self.noGradIdxs = []
        
        self.iterPerSample = 0
        
        self.mu = nn.Parameter(torch.FloatTensor([initialMu]),requires_grad = True)
        #self.m = nn.Parameter(torch.FloatTensor([initialm]),requires_grad = True)
        self.m = initialm
        #self.Q = nn.Linear(K,K, bias = False)
        self.Q = nn.Parameter(torch.zeros(K,K),requires_grad = False)
        self.pi = nn.Linear(K,1, bias = False)
        #self.NN = nn.Sequential(
        #                          nn.Linear(K,K)
        #                        )
        

        self.NLL_weight = 0.5
        self.pi_weight  = 0.5

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
        self.Q = nn.Parameter(torch.zeros(self.K,self.K),requires_grad = False)
        self.noGradIdxs = [] 
        for i in range(self.K):
            for j in range(self.K):
                if i == 0 and j == 0:
                    self.Q[i][j].data.clamp_(min = -inputLambda, max = -inputLambda)
                    self.noGradIdxs.append([i,j])
                elif i == j-1:
                    self.Q[i][j].data.clamp_(min = inputLambda, max = inputLambda)
                    self.noGradIdxs.append([i,j])
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
                            self.Q[i][j] = -1*(i-2)*self.mu
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
                    self.noGradIdxs.append([i,j])
                
                        
    
    def zero_out_gradients_for_nonParameters(self):
        for idx in self.noGradIdxs:
            self.Q.grad[idx[0]][idx[1]].fill_(0)   
                
        

    def toStr(self):
        s = 'asd'
        return s

    def paramClip(self):
        return
        self.m.data.clamp_(min=1.0, max=1e10)
        self.mu.data.clamp_(min=1.0, max=self.K-1)

    def forwardPass(self, inp):
        for i in range(self.iterPerSample):
            self.form_Q(inp)
            steady_state_loss = F.mse_loss( torch.mm(F.softmax(self.pi.weight, dim=1), self.Q), torch.zeros(self.pi.weight.size()) )
            #print '\tsteadyStateLoss=',steady_state_loss.item()
            steady_state_loss.backward(retain_graph=True)
            #self.mu.grad.fill_(0)
            nn.utils.clip_grad_norm_(self.parameters(), self.gradientClip, norm_type=self.gradientClipType)
            self.optimiser.step()
            self.optimiser.zero_grad()
            #print '\t', steady_state_loss.item()
            #print '\t', self.mu.item()
        probs = F.log_softmax(self.pi.weight, dim = 1)
        return probs[-1][-1]
        



    def objective(self, logPK, inputRates, actualDrops):
        # NLL = - [ (#dropped calls)*logPK + (#not dropped packets)*(1-logPK) ]

        NLL = -torch.sum(actualDrops * logPK + (inputRates - actualDrops) * torch.log(1 - torch.exp(logPK)))
        
        #P = F.softmax( self.NN(self.Q), dim = 1)
        #P = self.NN(self.Q)
        
        #steady_state_loss = F.mse_loss(torch.mm(F.softmax(self.pi.weight, dim=1), P), F.softmax(self.pi.weight, dim=1))
        steady_state_loss = F.mse_loss( torch.mm(F.softmax(self.pi.weight, dim=1), self.Q), torch.zeros(self.pi.weight.size()) )

        #obj = NLL
        obj = self.NLL_weight * NLL + self.pi_weight * steady_state_loss

        return obj
    
def evaluate(modelName, testLambdas, testPKs, plotFig = True):
    est_PKs = []
    for i in range(len(testLambdas)):
        testModel = torch.load(modelName)
        for e in range(10):
            testModel.form_Q(testLambdas[i])
            #P = F.softmax(testModel.NN(testModel.Q), dim = 1)
            #steady_state_loss = F.mse_loss(torch.mm(F.softmax(testModel.pi.weight, dim=1), P), F.softmax(testModel.pi.weight, dim=1))
            steady_state_loss = F.mse_loss( torch.mm(F.softmax(testModel.pi.weight, dim=1), testModel.Q), torch.zeros(testModel.pi.weight.size()) )
            #print steady_state_loss.item()
            steady_state_loss.backward()
            nn.utils.clip_grad_norm_(testModel.parameters(), testModel.gradientClip, norm_type=testModel.gradientClipType)
            testModel.mu.grad.fill_(0)
            #testModel.m.grad.fill_(0)
            testModel.optimiser.step()
            testModel.optimiser.zero_grad()
            
        logPK = testModel.forwardPass(testLambdas[i])
        est_PK = torch.exp(logPK).item()
        est_PKs.append(est_PK)
        #print est_PK, testPKs[i]
        
    est_PK_torch = torch.FloatTensor(est_PKs)
    true_PKs_torch = torch.FloatTensor(testPKs)
    MSE = F.mse_loss(est_PK_torch, true_PKs_torch)
    print 'Evaluation MSE=', MSE.item()
        
    if plotFig:  
        fig = plt.figure(1, figsize=(6, 4))
        axes = plt.gca()
        ax = plt.axes()
        
        #drawing est PK vs. emp PK
        plt.ylabel('Probability')
        plt.xlabel('Lambda')
        lines = plt.plot(testLambdas, est_PKs, '--r' ,label='Estimated PK') 
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(testLambdas, testPKs, 'b' ,label='Empirical PK') 
        plt.setp(lines, linewidth=2.0)
        plt.grid()                                                                     
        plt.show()
        
    return MSE.item()
    
        


def train(model = None, 
          inputLambdas = [], PKs = [], 
          batchSize=10, 
          epochs = 150,
          validLambdas = [], validPKs = [],
          modelName = 'abc'):
    
    model.train()
    bestValidMSE = 1e100
    bestModel = None
     
    total_loss = 0.0
    for e in range(epochs):
        est_PKs = []
        true_PKs = []
        for b in range(len(inputLambdas)):
            
            batch_X = inputLambdas[b]
            batch_Y = PKs[b] * inputLambdas[b]
            
            model.form_Q(batch_X)
            
            logPK = model.forwardPass(batch_X)
            
            est_PKs.append(torch.exp(logPK).item())
            true_PKs.append(PKs[b])
            
            loss = model.objective(logPK, batch_X, batch_Y)
            #loss.backward(retain_graph = True)
            total_loss = total_loss + loss
            if b % batchSize == 0:
                total_loss = total_loss / float(batchSize)
                total_loss.backward()
                #model.zero_out_gradients_for_nonParameters()
                nn.utils.clip_grad_norm_(model.parameters(), model.gradientClip, norm_type=model.gradientClipType)
                model.optimiser.step()
                model.optimiser.zero_grad()
                model.paramClip()
                #print 'Loss=', total_loss.item()
                #print 'mu=',model.mu.item()
                
                
                total_loss = 0.0
                
                
        est_PKs = torch.FloatTensor(est_PKs)
        true_PKs = torch.FloatTensor(true_PKs)
        MSE = F.mse_loss(est_PKs, true_PKs)
                
        torch.save(model, modelName)  
            
        validMSE = evaluate(modelName, validLambdas, validPKs, plotFig = False)
        
        if validMSE < bestValidMSE:
            bestValidMSE = validMSE
            bestModel = torch.load(modelName)
            
        print 'epoch=', e, ' Loss = ', loss.item(), 'lr=', model.lr
        print 'MSE=', MSE.item()
        print 'validMSE', validMSE
        print 'mu=',model.mu.item()
        #print 'pi=\n', model.pi.weight
        #print 'softmax(pi)=\n', F.softmax(model.pi.weight, dim=1)
        print '\tbest valid MSE', bestValidMSE
        print '\tbest mu=',bestModel.mu.item()
        #print '\t softmax(pi)=\n', F.softmax(bestModel.pi.weight, dim=1)
        print
    
    
    
    



def main():
    random.seed(1111)
    torch.manual_seed(1111)
    inputLambdas = list(range(10,500,1))
    random.shuffle(inputLambdas)
    
    true_m = 3
    true_mu = 50.0
    true_K = 5
    PKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K)) for inp in inputLambdas ]
    
    model = useGenericQueue_learn_MMmK(
        initialm = 2.0, 
        initialMu = 10.0, 
        K=true_K, lr=0.01, 
        gradientClip=0.25, gradientClipType=2, optim='Adam'
        )
    
    validLambdas = list(range(5,1000,7))
    validPKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K)) for inp in validLambdas ]
    
    train(model = model, 
          inputLambdas = inputLambdas, PKs = PKs, 
          batchSize = 10, 
          epochs = 150,
          validLambdas = validLambdas, validPKs = validPKs,
          modelName = 'abc')
    
    
    
    testLambdas = list(range(15,1000,10))
    testPKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K)) for inp in testLambdas ]
    
     
    evaluate('abc', testLambdas, testPKs)
    


if __name__ == "__main__":
    main()
    print('DONE!')