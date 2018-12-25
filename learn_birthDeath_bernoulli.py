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

class learn_birthDeath_bernoulli(nn.Module):
    def __init__(self, K = 10, lr = 0.01, gradientClip = 10.0, gradientClipType = 'inf', optim = 'SGD'):
        super(learn_birthDeath_bernoulli, self).__init__()
        #K has to be greater than m for the queueing eqn to work (K>m)
        
        self.K = K
        self.default_m = 5.0

        '''
        self.moduleList = nn.ModuleList()
        self.mus = {}
        for i in range(K):
            self.moduleList.append(nn.Parameter(torch.FloatTensor([self.default_m]),requires_grad = True))
            self.mus[i+1] = self.moduleList[-1]
        '''

        self.mus = nn.Parameter(torch.FloatTensor([self.default_m]*self.K),requires_grad = True)
        
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
        s = 'asd'
        return s
        
    def paramClip(self):
        pass

    
    def forwardPass(self, inp, clampNumbers = False):
        sums = 0
        for k in range(self.K):
            muls = 1.0
            for i in range(k):
                muls = muls * inp/self.mus[i]
            sums += muls
        p0 = 1.0 / (1.0 + sums)

        rhos = 1.0
        for i in range(self.K):
            rhos = rhos * (inp/self.mus[i])

        PK = p0 * rhos

        PK = torch.clamp(PK, min=1e-20, max=1.0)

        logPK = torch.log(PK)



        return logPK



    
    def predict(self, inp, clampNumbers):
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
    
    

def evaluate(model, data_X, data_Y):
    model.eval()
    totalLoss = 0
    
    logPK = model.forwardPass(data_X)
    #PK = torch.exp(logPK)
    #PK = torch.clamp(PK, min=0, max=1.0)
    loss = bernoulli_NLL(logPK, data_X, data_Y)
    totalLoss = loss.item()
    return totalLoss


def bernoulli_NLL(logPK, inputRates, actualDrops):
    #NLL = - [ (#dropped calls)*logPK + (#not dropped packets)*(1-logPK) ] 
        
    NLL = -torch.sum(actualDrops*logPK + (inputRates-actualDrops)*torch.log(1-torch.exp(logPK)) )
    #NLL = NLL / len(inputRates)
    return NLL

def train(model, train_X, train_Y, valid_X, valid_Y, test_X, test_Y, epochs, batchsize, annelingAt = 20, shuffleData = False, showBatches = False):
    model.train()
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
            logPK = model.forwardPass(batch_X, clampNumbers = False)
            #PK = torch.exp(logPK)
            #PK = torch.clamp(PK, min=0, max = 1.0)
            
            loss = bernoulli_NLL(logPK, batch_X, batch_Y)
            loss.backward()
            totalLoss += loss.item()
        
            nn.utils.clip_grad_norm_(model.parameters(), model.gradientClip , norm_type = model.gradientClipType)
            model.optimiser.step()
            model.optimiser.zero_grad()
            model.paramClip()
            
            if showBatches:
                print 'Batch=',b, ' NLL = ', loss.item(), 'lr=',model.lr
                print 'm=',model.m.item(), 'K=',model.K.item(), 'mu=',model.mu.item()
                #print 'm=',model.m.grad.item(), 'K=',model.K.grad.item(), 'mu=',model.mu.grad.item()
                print
            
        
        #trainLoss = evaluate(model, train_X, train_Y)
        validLoss = evaluate(model, valid_X, valid_Y) / float(len(valid_X))
        if len(test_X) != 0:
            testLoss = evaluate(model, test_X, test_Y) / float(len(test_X))
        else:
            testLoss = -1
            
        trainLoss = totalLoss / float(len(train_X))
        print 'Epoch',e,'--------------------------------------'
        print 'Total Train MSE loss = ', trainLoss, 'lr=', model.lr
        print 'Total Valid MSE loss = ', validLoss
        print 'Total Test MSE loss  = ', testLoss 
        print 'm=',model.m.item(), 'K=',model.K.item(), 'mu=',model.mu.item()
        print '--------------------------------------'
        
        #if validLoss > prevValidLoss:
        if model.optim == 'SGD' and e % annelingAt == 0 and e != 0:
            prevValidLoss = validLoss
            model.lr /= 2.0
            for param_group in model.optimiser.param_groups:
                param_group['lr'] = model.lr
            
        prevValidLoss = validLoss
        
        if (validLoss < bestValidLoss) or (validLoss == bestValidLoss and trainLoss < train_bestValidLoss) :
            bestValidLoss = validLoss 
            bestModel = copy.deepcopy(model)
            test_bestValidLoss = testLoss
            train_bestValidLoss = trainLoss
            #params = '_m=%0.3f_K=%0.3f_mu=%0.3f'%(bestModel.m.item(), bestModel.K.item(), bestModel.mu.item()) 
            torch.save(bestModel, 'model_model_'+model.toStr())
            
        print '\t', 'train loss=',train_bestValidLoss, 'best valid=',bestValidLoss, 'test loss=', test_bestValidLoss, ' m=',bestModel.m.item(), 'K=',bestModel.K.item(), 'mu=',bestModel.mu.item()
        
    torch.save(bestModel, 'birthDeath_model')
            







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
    
    model = learn_birthDeath_bernoulli(K = 10, lr=0.01, gradientClip = 0.25, gradientClipType = 2, optim = 'Adam')
    
    train(model, 
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