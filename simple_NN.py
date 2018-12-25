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
from collections import OrderedDict

class NN(nn.Module):
    def __init__(self, layerSizes, lr, gradientClip = 10.0, gradientClipType = 'inf'):
        super(NN, self).__init__()
        
        d = OrderedDict()
        d['l1'] = nn.Linear(1, layerSizes[0])
        d['a1'] = nn.ReLU()
        for i in range(len(layerSizes) - 1):
            d['l' + str(i + 2)] = nn.Linear(layerSizes[i], layerSizes[i + 1])
            d['a' + str(i + 2)] = nn.ReLU()
        
        d['l'+str(len(layerSizes)+1)] = nn.Linear(layerSizes[-1], 1)
        self.layers = nn.Sequential(d)
        
        
        self.loss = nn.MSELoss()
        self.lr = lr
        self.gradientClip = gradientClip
        self.gradientClipType = gradientClipType
        self.optimiser = torch.optim.SGD(self.parameters(), self.lr)
        paramsCount=0
        for p in list(self.parameters()):
            if len(p.shape) == 1:
                paramsCount += p.shape[0]
            else:
                paramsCount += p.shape[0] * p.shape[1]
        
        print '**************************************************' 
        print 'Number of parameters in the model = ',paramsCount
        print '**************************************************'
        
    
    
    def forwardPass(self, inp):
        return self.layers(inp.view(1,1))[0]
        '''
        final_outputs = []
        for i in range(inp.size(0)):
            sample = inp[i]
            pred = self.layers(sample.view(1,1))
            final_outputs.append(pred)
        final_outputs = Variable(torch.FloatTensor(final_outputs))
        return final_outputs
        '''
    def predict(self, inp):
        return self.forwardPass(torch.FloatTensor([inp])).item()


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
                PK = float(curr_row['FailedCall(P)'])/float(curr_row['CallRate(P)'])
            except:
                continue
            train_X.append(Variable(torch.FloatTensor([the_lambda])))
            train_Y.append(Variable(torch.FloatTensor([PK])))
            
    return train_X, train_Y
    
def evaluate(model, data_X, data_Y):
    model.eval()
    totalLoss = 0
    for i in range(len(data_X)):
        pred = model.forwardPass(data_X[i])
        loss = model.loss(pred, data_Y[i])
        totalLoss += loss.item()
    return totalLoss 

def train(model, train_X, train_Y, valid_X, valid_Y, test_X, test_Y, epochs, batchsize, annelingAt = 100, shuffleData = False):
    model.train()
    
    train_X = Variable(torch.FloatTensor(train_X))
    train_Y = Variable(torch.FloatTensor(train_Y))
    
    valid_X = Variable(torch.FloatTensor(valid_X))
    valid_Y = Variable(torch.FloatTensor(valid_Y))
    
    test_X = Variable(torch.FloatTensor(test_X))
    test_Y = Variable(torch.FloatTensor(test_Y))
    
    prevValidLoss = float('inf')
    bestValidLoss = float('inf')
    test_bestValidLoss = float('inf')
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
            batch_X = train_X[b:b+batchsize]
            batch_Y = train_Y[b:b+batchsize]
            batchloss = 0
            for i in range(len(batch_X)):
                PK = model.forwardPass(batch_X[i])
                loss = model.loss(PK, batch_Y[i])
                batchloss += loss
            totalLoss += batchloss.item()
            batchloss /= float(len(batch_X))
            batchloss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), model.gradientClip , norm_type = model.gradientClipType)
            model.optimiser.step()
            model.optimiser.zero_grad()
            #print 'Batch=',b, ' MSE loss = ', loss.item(), 'lr=',model.lr
            #print 'm=',model.m.item(), 'K=',model.K.item(), 'mu=',model.mu.item()
            #model.paramClip()
            #print 'm=',model.m.item(), 'K=',model.K.item(), 'mu=',model.mu.item()
        
        validLoss = evaluate(model, valid_X, valid_Y) / float(len(valid_X))
        if len(test_X) != 0:
            testLoss = evaluate(model, test_X, test_Y) / float(len(test_X))
        else:
            testLoss = -1
        print 'Epoch',e,'--------------------------------------'
        print 'Total Train MSE loss = ', totalLoss/float(len(train_X)), 'lr=', model.lr
        print 'Total Valid MSE loss = ', validLoss
        print 'Total Test MSE loss  = ', testLoss 
        print '--------------------------------------'
        if e % annelingAt == 0 and e != 0:
            prevValidLoss = validLoss
            model.lr /= 2.0
            for param_group in model.optimiser.param_groups:
                param_group['lr'] = model.lr
                
        prevValidLoss = validLoss
        
        if validLoss < bestValidLoss:
            bestValidLoss = validLoss 
            bestModel = model
            torch.save(bestModel, 'simpleNN_model')
            
            
        print '\t best valid=',bestValidLoss, 'test loss=', testLoss
            
            
        
    torch.save(bestModel, 'simpleNN_model')






def main():
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    summaryFile = 'summary_data_dump.csv'
    random.seed(1111)
    torch.manual_seed(1111)
    trainQuota = 0.85
    validQuota = 0.15
    
    data_X, data_Y = getTrainingData(dir, summaryFile, minDropRate=1, maxDropRate=100)
    
    #shuffle data
    #combined = list(zip(data_X, data_Y))
    #random.shuffle(combined)
    #data_X[:], data_Y[:] = zip(*combined)
    
    trainLen = int(trainQuota*len(data_X))
    validLen = int(validQuota*len(data_X))
    
    train_X = data_X[:trainLen]
    train_Y = data_Y[:trainLen]
    
    valid_X = data_X[trainLen:trainLen+validLen]
    valid_Y = data_Y[trainLen:trainLen+validLen]
    
    test_X = data_X[trainLen+validLen:]
    test_Y = data_Y[trainLen+validLen:]
    
    mynet = NN([10, 100, 10], lr=20.0, gradientClip = 0.25, gradientClipType = 2)
    
    train(mynet, train_X, train_Y,
          valid_X, valid_Y, 
          test_X, test_Y,
          10000, 32, annelingAt = 10,
          shuffleData = False
          )


if __name__ == "__main__":
    main()
    print('DONE!')