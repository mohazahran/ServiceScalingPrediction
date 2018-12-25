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
        self.power_P = 2
        
        self.iterPerSample = 100
        
        self.mu = nn.Parameter(torch.FloatTensor([initialMu]),requires_grad = True)
        self.m = initialm
        self.Q = torch.zeros(K,K)
        #self.Q = nn.Parameter(torch.zeros(K,K),requires_grad = True)
        self.pi = nn.Parameter(torch.FloatTensor([[1.0/K]*K]), requires_grad = True)
        #self.pi = torch.FloatTensor([[1.0/K]*K])
        #self.NN = nn.Sequential(
        #                          nn.Linear(K,K)
        #                        )
        

        self.NLL_weight = 1.0
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
        self.pi.data = torch.FloatTensor([list(range(self.K,0,-1))])
        for i in range(self.iterPerSample):
            self.form_Q(inputLa)
            #g = torch.max(torch.abs(self.Q))
            g = torch.max ( torch.diag(torch.ones(self.K)) * torch.abs(self.Q) )
            P = torch.diag(torch.ones(self.K)) + torch.div(self.Q, g)
            for t in range(self.power_P):
                P = torch.mm(P,P)
            pi_probs = F.softmax(self.pi, dim=1)
            pi_P = torch.mm(pi_probs, P)
            steady_state_loss = F.mse_loss( pi_P, pi_probs)
            #print '\tsteadyStateLoss=',steady_state_loss.item()
            
            steady_state_loss.backward()
            #steady_state_loss.backward(retain_graph=True)
            self.zero_out_gradients_for_nonParameters()
            self.mu.grad.fill_(0)
            nn.utils.clip_grad_norm_(self.parameters(), self.gradientClip, norm_type=self.gradientClipType)
            self.optimiser.step()
            self.optimiser.zero_grad()
        
        
        self.form_Q(inputLa)
        g = torch.max ( torch.diag(torch.ones(self.K)) * torch.abs(self.Q) )
        P = torch.diag(torch.ones(self.K)) + torch.div(self.Q, g)
        
        logPK = F.log_softmax(torch.mm(F.softmax(self.pi, dim=1), P), dim = 1)[-1][-1]
        
        # NLL = - [ (#dropped calls)*logPK + (#not dropped packets)*(1-logPK) ]
        #NLL = -torch.sum(dropCount * logPK + (inputLa - dropCount) * torch.log(1 - torch.exp(logPK)))
        NLL = F.mse_loss(torch.exp(logPK), torch.FloatTensor([dropProb]))
        
        steady_state_loss = F.mse_loss( torch.mm(F.softmax(self.pi, dim=1), P), F.softmax(self.pi, dim=1))
        
        loss = self.NLL_weight * NLL + self.steadyState_weight * steady_state_loss

        return loss, logPK
    
def evaluate(modelName, testLambdas, testPKs, plotFig = True):
    est_PKs = []
    avg_NLL = 0.0
    for i in range(len(testLambdas)):
        testModel = torch.load(modelName)
        testModel.pi.data = torch.FloatTensor([list(range(testModel.K,0,-1))])
        
        for e in range(testModel.iterPerSample):
            testModel.form_Q(testLambdas[i])
            #g = torch.max(torch.abs(testModel.Q))
            g = torch.max ( torch.diag(torch.ones(testModel.K)) * torch.abs(testModel.Q) )
            P = torch.diag(torch.ones(testModel.K)) + torch.div(testModel.Q, g)
            for t in range(testModel.power_P):
                P = torch.mm(P,P)
            pi_probs = F.softmax(testModel.pi, dim=1)
            pi_P = torch.mm(pi_probs, P)
            steady_state_loss = F.mse_loss( pi_P, pi_probs)
            steady_state_loss.backward()
            nn.utils.clip_grad_norm_(testModel.parameters(), testModel.gradientClip, norm_type=testModel.gradientClipType)
            testModel.mu.grad.fill_(0)
            testModel.optimiser.step()
            testModel.optimiser.zero_grad()
            
        logPK = F.log_softmax(torch.mm(F.softmax(testModel.pi, dim=1), P), dim = 1)[-1][-1]
        est_PK = torch.exp(logPK).item()
        est_PKs.append(est_PK)
        actualDrops = testPKs[i] * testLambdas[i]
        NLL = -torch.sum(actualDrops * logPK + (testLambdas[i] - actualDrops) * torch.log(1 - torch.exp(logPK)))
        avg_NLL += NLL.item()
        
        
    est_PK_torch = torch.FloatTensor(est_PKs)
    true_PKs_torch = torch.FloatTensor(testPKs)
    MSE = F.mse_loss(est_PK_torch, true_PKs_torch)
    avg_NLL = avg_NLL / float(len(testLambdas))
    #print 'Evaluation MSE=', MSE.item()
        
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
        
    return MSE.item(), avg_NLL
    
        


def train(model = None, 
          inputLambdas = [], PKs = [], 
          batchSize=10, 
          epochs = 150,
          validLambdas = [], validPKs = [],
          modelName = 'abc'):
    
    model.train()
    bestValidMSE = 1e100
    bestValidNLL = 1e100
    bestModel = None
     
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
            
            #loss.backward(retain_graph = True)
            total_loss = total_loss + loss
            epochLoss += loss.item()
            if b % batchSize == 0:
                total_loss = total_loss / float(batchSize) 
                total_loss.backward()
                #total_loss.backward(retain_graph=True)
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
            
        validMSE, validNLL = evaluate(modelName, validLambdas, validPKs, plotFig = False)
        
        if validNLL < bestValidNLL:
            bestValidMSE = validMSE
            bestValidNLL = validNLL
            bestModel = torch.load(modelName)
            
        print 'epoch=', e , 'lr=', model.lr, 'mu=',model.mu.item()
        print 'Loss= ', epochLoss/float(len(inputLambdas))
        print 'MSE=', MSE.item()
        print 'validMSE=', validMSE
        print 'validNLL=', validNLL
        #print 'pi=\n', model.pi.weight
        #print 'softmax(pi)=\n', F.softmax(model.pi.weight, dim=1)
        print '\tbest valid MSE', bestValidMSE
        print '\tbest validNLL=', bestValidNLL
        print '\tbest mu=',bestModel.mu.item()
        #print '\t softmax(pi)=\n', F.softmax(bestModel.pi.weight, dim=1)
        print
        
        torch.save(bestModel, modelName)
    
    
    



def main():
    random.seed(1111)
    torch.manual_seed(1111)
    inputLambdas = list(range(10,100,2))
    random.shuffle(inputLambdas)
    
    true_m = 3
    true_mu = 100.0
    true_K = 5
    
    #PKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K)) for inp in inputLambdas ]
    #PKs = [ solver.M_M_m_K_getProbAt_k_2(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in inputLambdas ]
    PKs = [ solver.M_M_m_K_getProbAt_k(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in inputLambdas ]
    
    
    model = useGenericQueue_learn_MMmK(
        initialm = 3.0, 
        initialMu = 25.0, 
        K=true_K, 
        lr=0.1, gradientClip=0.5, gradientClipType=2, optim='Adam'
        )
    
    validLambdas = list(range(5,100,7))
    validPKs = [ solver.M_M_m_K_getProbAt_k(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in validLambdas ]
    
    train(model = model, 
          inputLambdas = inputLambdas, PKs = PKs, 
          batchSize = 1, 
          epochs = 25,
          validLambdas = validLambdas, validPKs = validPKs,
          modelName = 'abc')
    
    
    
    testLambdas = list(range(15,100,10))
    testPKs = [ solver.M_M_m_K_getProbAt_k(float(inp)/float(true_mu), true_m, true_K, true_K) for inp in testLambdas ]
    
     
    evaluate('abc', testLambdas, testPKs)
    


if __name__ == "__main__":
    main()
    print('DONE!')