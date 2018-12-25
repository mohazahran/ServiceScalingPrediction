'''
Created on Oct 27, 2018

@author: mohame11
'''
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import random
import copy
import sys
import math

class learn_MMmK_bernoulli(nn.Module):
    def __init__(self, lr, gradientClip = 10.0, gradientClipType = 'inf', optim = 'SGD'):
        super(learn_MMmK_bernoulli, self).__init__()
        #K has to be greater than m for the queueing eqn to work (K>m)
        self.default_m = 1.0
        self.default_K = 5.0
        self.default_mu = 5.0
        
        self.m_min,   self.m_max = 1.0, 1e10
        self.K_min,   self.K_max = 5.0, 1e10
        self.mu_min, self.mu_max = 5.0, 1e10
        
        self.clampMin = -1e37
        self.clampMax = 1e37
        
        self.m = nn.Parameter(torch.FloatTensor([self.default_m]),requires_grad = True)
        self.mu = nn.Parameter(torch.FloatTensor([self.default_mu]),requires_grad = True)
        self.K = nn.Parameter(torch.FloatTensor([self.default_K]),requires_grad = True)
        
        self.lr = lr
        self.gradientClip = gradientClip
        self.gradientClipType = gradientClipType
        self.optim = optim
        self.optimiser = getattr(torch.optim, self.optim)(self.parameters(), lr = self.lr)
        #self.optimiser = torch.optim.SGD(self.parameters(), self.lr)
        paramsCount=0
        for p in list(self.parameters()):
            if len(p.shape) == 1:
                paramsCount += p.shape[0]
            else:
                paramsCount += p.shape[0] * p.shape[1]
        
        print '**************************************************' 
        print 'Number of parameters in the model = ',paramsCount
        print '**************************************************'
        
    def toStr(self):
        #s = 'm0=%.1f_K0=%.1f_mu0=%.1f_mixed_.5_0_.5_model_S'%(self.default_m, self.default_K, self.default_mu)
        s = 'asd'
        return s
        
    def paramClip(self):
        
        #if self.m.item() <= 0:
        self.m.data.clamp_(min=self.m_min, max=self.m_max)
            #self.m = nn.Parameter(torch.FloatTensor([self.default_m]),requires_grad = True)
            
        #if self.K.item() <= 0:
        self.K.data.clamp_(min=self.K_min, max=self.K_max)
            #self.K = nn.Parameter(torch.FloatTensor([self.default_K]),requires_grad = True)
        
        #if self.mu.item() <= 0:
        self.mu.data.clamp_(min=self.mu_min, max=self.mu_max)
            #self.mu = nn.Parameter(torch.FloatTensor([self.default_K_mu]),requires_grad = True)
        
    def forwardPass_new(self, inp):
        rho = torch.div(inp, self.mu)
        #x = inp / self.mu
        c1 = self.K * torch.log(rho)
        c2 = torch.lgamma(self.m + 1)
        c3 = (self.K - self.m) * torch.log(self.m)
        logC_K = c1 - c2 - c3
        
        
        part2 = torch.zeros(rho.size())
        for n in range(1,self.m):
            f1 = torch.pow(rho , n)
            f1 = torch.clamp(f1, min=self.min, max=self.max)
            f2 = torch.exp(torch.lgamma(torch.FloatTensor([n+1])))
            f2 = torch.clamp(f2, min=self.min, max=self.max)
            f = f1 / f2
            f = torch.clamp(f, min=self.min, max=self.max)
            part2 += f
            part2 = torch.clamp(part2, min=self.min, max=self.max)
            
            
        part2 = torch.clamp(part2, min=self.min, max=self.max)
        
        '''  
        part3 = torch.zeros(rho.size())
        for n in range(self.m, self.K+1):
            part3 +=  torch.exp( (n * torch.log(rho)) - ((n-self.m) * torch.log(self.m)) )
        part3 = part3 / (torch.exp(torch.lgamma(self.m+1)))
        '''
        
        
         
        part3 = torch.exp((self.K+1) * torch.log(rho))
        part3 = torch.clamp(part3, min=self.min, max=self.max)
        
        p = torch.exp((self.m - self.K) * torch.log(self.m))
        p = torch.clamp(p, min=self.min, max=self.max)
        part3 = part3 * p
        part3 = torch.clamp(part3, min=self.min, max=self.max)
        
        p = (self.m * torch.exp(self.m * torch.log(rho)))
        p = torch.clamp(p, min=self.min, max=self.max)
        part3 = part3 - p
        part3 = torch.clamp(part3, min=self.min, max=self.max)
        
        p = (self.m - rho)
        p = torch.clamp(p, min=self.min, max=self.max)
        part3 = part3 / p
        part3 = torch.clamp(part3, min=self.min, max=self.max)
        
        p = (torch.exp(torch.lgamma(self.m+1)))
        p = torch.clamp(p, min=self.min, max=self.max)
        part3 = part3 / p
        part3 = torch.clamp(part3, min=self.min, max=self.max)
        
        part3 = part3 * (-1)
        part3 = torch.clamp(part3, min=self.min, max=self.max)
        
        #part3a = (self.m-self.K)*torch.log(self.m) + (self.K+1) * torch.log(rho) - torch.log(rho - self.m) - torch.lgamma(self.m+1)
        #part3b = torch.log(self.m) + (self.m * torch.log(rho)) - torch.log(self.m-rho) - torch.lgamma(self.m+1)
        #part3 = torch.exp(part3a) + torch.exp(part3b)
        
        p = 1.0 + part2 + part3
        p = torch.clamp(p, min=1.0/1e25, max=self.max)
        logP0 = -1 * torch.log(p)
        
        logP_K = logC_K + logP0
        
        return logP_K
        
    
    def forwardPass(self, inp, clampNumbers = False):
        
        #cmin = self.clampMin
        #cmax = self.clampMax
        cmin = -1e10
        cmax = 1e10
        zeroClose = 1e-10
        
        rho = torch.div(inp, self.mu)
        #x = inp / self.mu
        if clampNumbers:
            rho = torch.clamp(rho, min = zeroClose, max = cmax)
        c1 = self.K * torch.log(rho)
        c2 = torch.lgamma(self.m + 1)
        c3 = (self.K - self.m) * torch.log(self.m)
        logC_K = c1 - c2 - c3
        if clampNumbers:
            logC_K = torch.clamp(logC_K, min = self.clampMin, max = 1.0)
        
        part2 = torch.zeros(rho.size())
        for n in range(1,self.m):
            if clampNumbers:
                n1 = torch.pow(rho , n)
                n1 = torch.clamp(n1, min=cmin, max=cmax)
                n2 = torch.exp(torch.lgamma(torch.FloatTensor([n+1])))
                n2 = torch.clamp(n2, min=cmin, max=cmax)
                part2 = part2 + (n1/n2)
                part2 = torch.clamp(part2, min = cmin, max = cmax) 
            else:
                part2 = part2 + torch.pow(rho , n) / torch.exp(torch.lgamma(torch.FloatTensor([n+1])))
        
        '''
        part3 = torch.zeros(rho.size())
        for n in range(self.m, self.K+1):
            part3 +=  torch.clamp ( torch.exp( (n * torch.log(rho)) - ((n-self.m) * torch.log(self.m)) ) , min = self.clampMin, max = self.clampMax)
            part3 = torch.clamp(part3, min = self.clampMin, max = self.clampMax)
        part3 = part3 / (torch.exp(torch.lgamma(self.m+1)))
        '''
            
        
        part3 = torch.zeros(rho.size())
        for n in range(self.m, self.K+1):
            if clampNumbers:
                a = torch.exp( (n * torch.log(rho)) - ((n-self.m) * torch.log(self.m)) )
                a = torch.clamp(a, min = cmin, max = cmax)
                part3 = part3 + a
                part3 = torch.clamp(part3, min = cmin, max = cmax)
            else:
                part3 =  part3 + torch.exp( (n * torch.log(rho)) - ((n-self.m) * torch.log(self.m)) )
                
        if clampNumbers:
            part3 = part3 / (torch.exp(torch.lgamma(self.m+1)))
            part3 = torch.clamp(part3, min = cmin, max = cmax)
        else:
            part3 = part3 / (torch.exp(torch.lgamma(self.m+1)))
        
        
        '''
        #default way of cal part3 (it has a mistake of forgetting to divide by m! in the end
        part3 = torch.zeros(rho.size())
        for n in range(self.m, self.K+1):
            part3 += torch.pow(rho , n) / torch.pow(self.m,(n - self.m))
        '''    
        
            
        logP0 = -1 * torch.log(1.0 + part2 + part3)
        if clampNumbers:
            logP0 = torch.clamp(logP0, min = self.clampMin, max = 1.0)
        
        logP_K = logC_K + logP0
        
        if clampNumbers:
            logP_K = torch.clamp(logP_K, min = self.clampMin, max = 1.0)
        
        return logP_K
    
    def predict(self, inp, clampNumbers):
        clampNumbers = False
        logPK = self.forwardPass(torch.FloatTensor([inp]), clampNumbers)
        if clampNumbers:
            PK = torch.exp(logPK)
            PK = torch.clamp(PK, min = 0.0, max = 1.0)
        else:
            PK = torch.exp(logPK)
            
        return PK.item()
        
        


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
    
    

def evaluate(MMmK, data_X, data_Y):
    MMmK.eval()
    totalLoss = 0
    
    logPK = MMmK.forwardPass(data_X)
    #PK = torch.exp(logPK)
    #PK = torch.clamp(PK, min=0, max=1.0)
    loss = bernoulli_NLL(logPK, data_X, data_Y)
    totalLoss = loss.item()
    return totalLoss


def bernoulli_NLL(logPK, inputRates, actualDrops):
    #NLL = - [ (#dropped calls)*logPK + (#not dropped packets)*(1-logPK) ] 
        
    NLL = -torch.sum(actualDrops*logPK + (inputRates-actualDrops)*torch.log(1-torch.exp(logPK)) )
    NLL = NLL / float(len(inputRates))
    return NLL

def train(MMmK, train_X, train_Y, valid_X, valid_Y, test_X, test_Y, epochs, batchsize, annelingAt = 20, shuffleData = False, showBatches = False):
    MMmK.train()
    train_X = Variable(torch.FloatTensor(train_X))
    train_Y = Variable(torch.FloatTensor(train_Y))
    
    valid_X = Variable(torch.FloatTensor(valid_X))
    valid_Y = Variable(torch.FloatTensor(valid_Y))
    
    test_X = Variable(torch.FloatTensor(test_X))
    test_Y = Variable(torch.FloatTensor(test_Y))
    
    prevValidLoss = 1e100
    bestValidLoss = 1e100
    test_bestValidLoss = 1e100
    train_bestValidLoss = 1e100
    bestModel = None
    numOfBatchs = train_X.size(0) // batchsize
    for e in range(epochs):
        totalLoss = 0
        
        #shuffle training data
        if shuffleData:
            rows = len(train_X)
            idxs = list(range(0,rows))
            random.shuffle(idxs)
            idxs = torch.LongTensor(idxs)
            train_X = train_X[idxs]
            train_Y = train_Y[idxs]
        
        for b in range(numOfBatchs):
            if b == 167:
                dbg = 1
            batch_X = train_X[b:b+batchsize] 
            batch_Y = train_Y[b:b+batchsize] 
            logPK = MMmK.forwardPass(batch_X, clampNumbers = False)
            #PK = torch.exp(logPK)
            #PK = torch.clamp(PK, min=0, max = 1.0)
            
            loss = bernoulli_NLL(logPK, batch_X, batch_Y)
            loss.backward()
            totalLoss += loss.item()
        
            nn.utils.clip_grad_norm_(MMmK.parameters(), MMmK.gradientClip , norm_type = MMmK.gradientClipType)
            MMmK.optimiser.step()
            MMmK.optimiser.zero_grad()
            MMmK.paramClip()
            
            if showBatches:
                print 'Batch=',b, ' NLL = ', loss.item(), 'lr=',MMmK.lr
                print 'm=',MMmK.m.item(), 'K=',MMmK.K.item(), 'mu=',MMmK.mu.item()
                #print 'm=',MMmK.m.grad.item(), 'K=',MMmK.K.grad.item(), 'mu=',MMmK.mu.grad.item()
                print
            
        
        #trainLoss = evaluate(MMmK, train_X, train_Y)
        validLoss = evaluate(MMmK, valid_X, valid_Y) / float(len(valid_X))
        if len(test_X) != 0:
            testLoss = evaluate(MMmK, test_X, test_Y) / float(len(test_X))
        else:
            testLoss = -1
            
        trainLoss = totalLoss / float(len(train_X))
        print 'Epoch',e,'--------------------------------------'
        print 'Total Train MSE loss = ', trainLoss, 'lr=', MMmK.lr
        print 'Total Valid MSE loss = ', validLoss
        print 'Total Test MSE loss  = ', testLoss 
        print 'm=',MMmK.m.item(), 'K=',MMmK.K.item(), 'mu=',MMmK.mu.item()
        print '--------------------------------------'
        
        #if validLoss > prevValidLoss:
        if MMmK.optim == 'SGD' and e % annelingAt == 0 and e != 0:
            prevValidLoss = validLoss
            MMmK.lr /= 2.0
            for param_group in MMmK.optimiser.param_groups:
                param_group['lr'] = MMmK.lr
            
        prevValidLoss = validLoss
        
        if (validLoss < bestValidLoss) or (validLoss == bestValidLoss and trainLoss < train_bestValidLoss) :
            bestValidLoss = validLoss 
            bestModel = copy.deepcopy(MMmK)
            test_bestValidLoss = testLoss
            train_bestValidLoss = trainLoss
            #params = '_m=%0.3f_K=%0.3f_mu=%0.3f'%(bestModel.m.item(), bestModel.K.item(), bestModel.mu.item()) 
            torch.save(bestModel, 'MMmK_model_'+MMmK.toStr())
            
        print '\t', 'train loss=',train_bestValidLoss, 'best valid=',bestValidLoss, 'test loss=', test_bestValidLoss, ' m=',bestModel.m.item(), 'K=',bestModel.K.item(), 'mu=',bestModel.mu.item()
        
    torch.save(bestModel, 'MMmK_model_bernoulli')
            







def main():
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_4_K_DEFT_SCALE_86_2018.10.29-22.19.41/sipp_results/'
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_2_K_100000_SCALE_43_2018.11.03-13.38.21/sipp_results/'
    
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.4+.5_CORE_1_K_425982_SCALE_24_2018.11.23-18.47.32/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_INVITE_CORE_1_K_425982_SCALE_17_2018.11.21-02.49.02/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_MSG_CORE_1_K_425982_SCALE_80_2018.11.21-04.56.39/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_SUBSCRIBE_CORE_1_K_425982_SCALE_36_2018.11.21-07.04.10/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.5+0_CORE_1_K_425982_SCALE_30_2018.11.23-20.55.11/sipp_results/'
    
    summaryFile = 'summary_data_dump.csv'
    
    
    random.seed(1111)
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
    
    MMmK = learn_MMmK_bernoulli(lr=0.01, gradientClip = 0.25, gradientClipType = 2, optim = 'Adam')
    
    train(MMmK, 
          train_X, train_Y,
          valid_X, valid_Y, 
          test_X, test_Y,
          epochs = 10000, batchsize = 32, annelingAt = 500,
          shuffleData = True,
          showBatches = False
          )
    
    


if __name__ == "__main__":
    main()
    print('DONE!')