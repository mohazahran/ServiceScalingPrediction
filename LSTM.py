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
from collections import OrderedDict

CUDA = False

class MyLSTM(nn.Module):
    def __init__(self, lstmHiddenSize, layerSizes, numLayers, win, optim = 'SGD', lr = 0.01, gradientClip = 10.0, gradientClipType = 'inf'):
        super(MyLSTM, self).__init__()
        
        
        self.lstm = nn.LSTM(input_size=win,
                           hidden_size=lstmHiddenSize,
                           num_layers=numLayers,
                           bias=True)
        
        d = OrderedDict()
        d['l1'] = nn.Linear(lstmHiddenSize, layerSizes[0])
        d['a1'] = nn.ReLU()
        #d['a1'] = nn.ELU()
        for i in range(len(layerSizes) - 1):
            d['l' + str(i + 2)] = nn.Linear(layerSizes[i], layerSizes[i + 1])
            d['a' + str(i + 2)] = nn.ReLU()
            #d['a' + str(i + 2)] = nn.ELU()
        
        d['l'+str(len(layerSizes)+1)] = nn.Linear(layerSizes[-1], 1)
        self.layers = nn.Sequential(d)
        
        #self.hiddenLayer = nn.Linear(hiddenSize, 1)
        self.optim = optim
        self.hiddenSize = lstmHiddenSize
        self.history = []
        self.win = win
        self.loss = nn.MSELoss()
        self.lr = lr
        self.gradientClip = gradientClip
        self.gradientClipType = gradientClipType
        self.optimiser = getattr(torch.optim, self.optim)(self.parameters(), lr = self.lr)
        paramsCount=0
        for p in list(self.parameters()):
            if len(p.shape) == 1:
                paramsCount += p.shape[0]
            else:
                paramsCount += p.shape[0] * p.shape[1]
        
        print '**************************************************' 
        print 'Number of parameters in the model = ',paramsCount
        print '**************************************************'
        
        torch.set_num_threads(4)
    
    def forwardPass(self, inp):
        #LSTM input should be (seq_len, batch, input_size (emb size))
        inp = inp.view(inp.size(0), 1, 1)
        output, (h_n, c_n) = self.lstm(inp)
        
        #LSTM output is (seq_len, batch, num_directions * hidden_size)
        final_outputs = self.layers(output)
        
        return final_outputs.view(-1)
        
        '''
        final_outputs = []
        for i in range(inp.size(0)):
            sample = inp[i]
            pred = self.layers(sample.view(1,1))
            final_outputs.append(pred)
        final_outputs = Variable(torch.FloatTensor(final_outputs))
        return final_outputs
        '''
    def predict(self, x, clampNumbers = True):
        return self.forwardPass(x)
    
    def format_trainingData(self, data_X, data_Y):
        return self.batchfiy(data_X, data_Y)
    
    def batchfiy(self, train_X, train_Y):
        batches_X = []
        batches_Y = []
        bX = []
        bY = []
        for i in range(len(train_X)):
            if train_X[i] != -1:
                bX.append([train_X[i]])
                bY.append(train_Y[i])
            else:
                if len(bX) == 0:
                    continue 
                
                if CUDA:
                    bX = Variable(torch.FloatTensor(bX)).cuda()
                    bY = Variable(torch.FloatTensor(bY)).cuda()
                else:
                    bX = Variable(torch.FloatTensor(bX))
                    bY = Variable(torch.FloatTensor(bY))
                batches_X.append(bX)
                batches_Y.append(bY)
                bX = []
                bY = []
        return batches_X, batches_Y


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
            train_X.append(the_lambda)
            train_Y.append(PK)
        
        train_X.append(-1) #markers to flush states
        train_Y.append(-1)
            
    return train_X, train_Y
    
def evaluate(model, data_X, data_Y):
    model.eval()
    
    
    totalLoss = 0
    batches_X, batches_Y = batchfiy(data_X, data_Y)
    for b in range(len(batches_X)):
        batch_X = batches_X[b]
        batch_Y = batches_Y[b]
        
        pred = model.forwardPass(batch_X)
        loss = model.loss(pred, batch_Y)
        totalLoss += loss.item()
        
    return totalLoss / len(batches_X) 


    

def train(model, train_X, train_Y, valid_X, valid_Y, test_X, test_Y, epochs, batchsize, annelingAt = 100, shuffleData = False):
    model.train()
    
    batches_X, batches_Y = batchfiy(train_X,train_Y)
    #train_X = Variable(torch.FloatTensor(train_X))
    #train_Y = Variable(torch.FloatTensor(train_Y))
    
    #valid_X = Variable(torch.FloatTensor(valid_X))
    #valid_Y = Variable(torch.FloatTensor(valid_Y))
    
    #test_X = Variable(torch.FloatTensor(test_X))
    #test_Y = Variable(torch.FloatTensor(test_Y))
    
    prevValidLoss = 1e100
    bestValidLoss = 1e100
    test_bestValidLoss = 1e100
    train_bestValidLoss = 1e100
    bestModel = None
    
    #numOfBatchs = len(train_X) // batchsize
    for e in range(epochs):
        totalLoss = 0
        
        for b in range(len(batches_X)):
            batch_X = batches_X[b]
            batch_Y = batches_Y[b]
            
                    
            PK = model.forwardPass(batch_X)
            loss = model.loss(PK, batch_Y)
            totalLoss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), model.gradientClip , norm_type = model.gradientClipType)
            model.optimiser.step()
            model.optimiser.zero_grad()
            #print 'Batch=',b, ' MSE loss = ', loss.item(), 'lr=',model.lr
            #print 'm=',model.m.item(), 'K=',model.K.item(), 'mu=',model.mu.item()
            #model.paramClip()
            #print 'm=',model.m.item(), 'K=',model.K.item(), 'mu=',model.mu.item()
        
        if len(valid_X) > 1:
            validLoss = evaluate(model, valid_X, valid_Y)
        if len(test_X) > 1:
            testLoss = evaluate(model, test_X, test_Y)
        else:
            testLoss = -1
        trainLoss = totalLoss / float(len(batch_X))
        print 'Epoch',e,'--------------------------------------'
        print 'Total Train MSE loss = ', trainLoss, 'lr=', model.lr
        print 'Total Valid MSE loss = ', validLoss
        print 'Total Test MSE loss  = ', testLoss 
        print '--------------------------------------'
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
            torch.save(bestModel, 'LSTM_model')
            
            
        print '\t', 'train loss=',train_bestValidLoss, 'best valid=',bestValidLoss, 'test loss=', test_bestValidLoss
            
            
        
    torch.save(bestModel, 'LSTM')






def main():
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_4_K_DEFT_SCALE_86_2018.10.29-22.19.41/sipp_results/'
    summaryFile = 'summary_data_dump.csv'
    random.seed(1111)
    torch.manual_seed(1111)
    trainQuota = 0.95
    validQuota = 0.05
    
    
    data_X, data_Y = getTrainingData(dir, summaryFile, minDropRate=1, maxDropRate=1e100)
    
    #shuffle data
    #combined = list(zip(data_X, data_Y))
    #random.shuffle(combined)
    #data_X[:], data_Y[:] = zip(*combined)
    
    trainLen = int(trainQuota*len(data_X))
    validLen = int(validQuota*len(data_X))
    
    print 'trainLen=', trainLen, 'validLen=', validLen
    
    train_X = data_X[:trainLen]
    train_Y = data_Y[:trainLen]
    
    valid_X = data_X[trainLen:trainLen+validLen]
    valid_Y = data_Y[trainLen:trainLen+validLen]
    
    test_X = data_X[trainLen+validLen:]
    test_Y = data_Y[trainLen+validLen:]
    
    mynet = MyLSTM(lstmHiddenSize = 50, layerSizes = [25,10], numLayers = 1, win = 1, optim = 'Adam', lr = 0.01, gradientClip = 50.0, gradientClipType = 2)
    if CUDA:
        mynet = mynet.cuda()
    
    train(mynet, train_X, train_Y,
          valid_X, valid_Y, 
          test_X, test_Y,
          10000, 32, annelingAt = 3,
          shuffleData = False
          )


if __name__ == "__main__":
    main()
    print('DONE!')