'''
Created on Nov 28, 2018

@author: mohame11
'''
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
from learn_MMmK import *

class MMmKs_mixTraffic(nn.Module):
    def __init__(self, models, lr, gradientClip = 10.0, gradientClipType = 'inf', optim = 'SGD'):
        super(MMmKs_mixTraffic, self).__init__()
        
        self.models = models
        
        '''
        self.sequential = nn.Sequential(
                                          nn.Linear(1, 10),
                                          nn.ReLU(),
                                          nn.Linear(10, len(models)),
                                          nn.ReLU()
                                          )
        '''
        
        self.inputSequential = nn.Sequential(
                                            nn.Linear(1, len(models), bias=False),
                                            #nn.Sigmoid(),
                                            #nn.Linear(5, len(models)),
                                            nn.ReLU(),
                                            )
        
        self.outputSequential = nn.Sequential(
                                            nn.Linear(len(models), 1, bias=False),
                                            #nn.Sigmoid(),
                                            #nn.Linear(5, 1),
                                            nn.ReLU(),
                                            )
        
        
        self.inputSequential[0].weight.data.fill_(1.0/len(models))
        self.outputSequential[0].weight.data.fill_(1.0/len(models))
        #self.linear = nn.Linear(1, len(models), bias=True)
        #self.linear.weight.data.fill_(1.0/len(models))
        
        #self.outLayer = nn.Linear(len(models),1, bias = True)
        #self.outLayer.weight.data.fill_(1.0/len(models))
        #self.linear.bias.data.fill_(1.0/len(models))
        
        self.loss = nn.MSELoss(size_average = True)
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
        s = 'learnt_trafficShares'
        return s
        
    
        
    
    def forwardPass(self, inp, clampNumbers = False):
        '''
        #[model_invite, model_msg, model_subsc]
        pred = 0
        inp = inp.view(inp.size(0),1)
        #scaledInp = self.sequential(inp)
        scaledInp = self.linear(inp)
        for i in range(scaledInp.size(1)):
            pred = pred + torch.exp(self.models[i].forwardPass(scaledInp[:,i]))
        return pred
        '''
        
        '''
        inp = inp.view(inp.size(0),1)
        scaledInp = self.linear(inp)
        pred = Variable(torch.zeros_like(scaledInp))
        for i in range(scaledInp.size(1)):
            pred[:,i] = torch.exp(self.models[i].forwardPass(scaledInp[:,i]))
        
        outp = self.outLayer(pred)
        #outp = torch.clamp(outp, min = 0.0, max = 1.0)
        return outp.view(outp.size(0))
        '''
    
        inp = inp.view(inp.size(0),1)
        scaledInp = self.inputSequential(inp)
        pred = Variable(torch.zeros_like(scaledInp))
        for i in range(scaledInp.size(1)):
            pred[:,i] = torch.exp(self.models[i].forwardPass(scaledInp[:,i]))
        
        outp = self.outputSequential(pred)
        #outp = torch.clamp(outp, min = 0.0, max = 1.0)
        return outp.view(outp.size(0))
        
    
    
            
            
    
        


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
                if failed > the_lambda:
                    continue
                PK = failed/the_lambda
            except:
                continue
            train_X.append(Variable(torch.FloatTensor([the_lambda])))
            train_Y.append(Variable(torch.FloatTensor([PK])))
            
    return train_X, train_Y
    
    

def evaluate(MMmK, data_X, data_Y):
    MMmK.eval()
    totalLoss = 0
    #for i in range(len(data_X)):
    PK = MMmK.forwardPass(data_X)
        #PK = torch.exp(logPK)
    PK = torch.clamp(PK, min=0, max=1.0)
    loss = MMmK.loss(PK, data_Y)
    totalLoss += loss.item()
    return totalLoss

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
            if b == 821:
                dbg = 1
            batch_X = train_X[b:b+batchsize]
            batch_Y = train_Y[b:b+batchsize]
            PK = MMmK.forwardPass(batch_X)
            #PK = torch.exp(logPK)
            #PK = torch.clamp(PK, min=0, max = 1.0)
            loss = MMmK.loss(PK, batch_Y)
            loss.backward()
            totalLoss += loss.item()*batchsize
        
            nn.utils.clip_grad_norm_(MMmK.parameters(), MMmK.gradientClip , norm_type = MMmK.gradientClipType)
            MMmK.optimiser.step()
            MMmK.optimiser.zero_grad()
            #MMmK.paramClip()
            
            if showBatches:
                print 'Batch=',b, ' MSE loss = ', loss.item(), 'lr=',MMmK.lr
                print 'm=',MMmK.m.item(), 'K=',MMmK.K.item(), 'mu=',MMmK.mu.item()
                print
            
        
        #trainLoss = evaluate(MMmK, train_X, train_Y)
        validLoss = evaluate(MMmK, valid_X, valid_Y) 
        if len(test_X) != 0:
            testLoss = evaluate(MMmK, test_X, test_Y) 
        else:
            testLoss = -1
            
        trainLoss = totalLoss / float(len(train_X))
        print 'Epoch',e,'--------------------------------------'
        print 'Total Train MSE loss = ', trainLoss, 'lr=', MMmK.lr
        print 'Total Valid MSE loss = ', validLoss
        print 'Total Test MSE loss  = ', testLoss 
        print 'input weights=', MMmK.inputSequential[0].weight.data
        print 'output weights=', MMmK.outputSequential[0].weight.data
        #print 'output weights=', MMmK.outLayer.weight.data
        #print  MMmK.linear.bias.data
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
            
        print '\t', 'train loss=',train_bestValidLoss, 'best valid=',bestValidLoss, 'test loss=', test_bestValidLoss
        print 'best input weights=', bestModel.inputSequential[0].weight.data
        print 'best input weights=', bestModel.outputSequential[0].weight.data
        #print 'best output weights=', bestModel.outLayer.weight.data
        #print 'bias=',bestModel.linear.bias.data
        
        #print '\t', 'best train loss=',train_bestValidLoss, 'best valid=',bestValidLoss, 'test loss=', test_bestValidLoss
        #print 'weights=', bestModel.linear.weight.data
        
    torch.save(bestModel, 'MMmK_model')
            







def main():
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_4_K_DEFT_SCALE_86_2018.10.29-22.19.41/sipp_results/'
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_2_K_100000_SCALE_43_2018.11.03-13.38.21/sipp_results/'
    
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.4+.5_CORE_1_K_425982_SCALE_24_2018.11.23-18.47.32/sipp_results/'
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.5+0_CORE_1_K_425982_SCALE_30_2018.11.23-20.55.11/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_INVITE_CORE_1_K_425982_SCALE_17_2018.11.21-02.49.02/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_MSG_CORE_1_K_425982_SCALE_80_2018.11.21-04.56.39/sipp_results/'
    
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_SUBSCRIBE_CORE_1_K_425982_SCALE_36_2018.11.21-07.04.10/sipp_results/'
    
    summaryFile = 'summary_data_dump.csv'
    
    
    random.seed(1111)
    trainQuota = 0.85
    validQuota = 0.15
    
    data_X, data_Y = getTrainingData(dir, summaryFile, minDropRate=1, maxDropRate=100)
    
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
    
    test_X, test_Y = getTrainingData(dir, summaryFile, minDropRate=-1, maxDropRate=1e100)
    
    
    model_invite = torch.load('MMmK_model_m0=1.0_K0=5.0_mu0=5.0_invite_model_L')
    model_msg = torch.load('MMmK_model_m0=1.0_K0=5.0_mu0=5.0_msg_model_O')
    model_subsc = torch.load('MMmK_model_m0=1.0_K0=5.0_mu0=5.0_subsc_model_P')
    
    models = [model_invite, model_msg, model_subsc]
    
    MMmK = MMmKs_mixTraffic(models, lr=0.01, gradientClip = 0.25, gradientClipType = 2, optim = 'Adam')
    
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