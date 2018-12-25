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


class useGenericQueue_learn_MMmK(nn.Module):
    def __init__(self, K=10, lr=0.01, gradientClip=10.0, gradientClipType='inf', optim='SGD'):
        super(useGenericQueue_learn_MMmK, self).__init__()
        # K has to be greater than m for the queueing eqn to work (K>m)

        #Define the markov chain
        '''
        self.P = [['0']*K for i in range(K)]
        for i in range(K):
            for j in range(K):
                if i == j:
                    if i < m:
                        if j-1 >=0:
                            self.P[i][j-1] =  str(i)+'mu'
                        if j+1 < K:
                            self.P[i][j+1] =  'lambda'
                            self.P[i][j]   = '-('+str(i)+'mu+lambda)'
                        else:
                            self.P[i][j]   = '-('+str(i)+'mu)'
                            
                            
                        
                    else:
                        if j-1 >=0:
                            self.P[i][j-1] = str(m)+'mu'
                        if j+1 < K:
                            self.P[i][j+1] =  'lambda'
                            self.P[i][j]   = '-('+str(m)+'mu+lambda)'
                        else:
                            self.P[i][j]   = '-('+str(m)+'mu)'
        '''
        
        
        
        '''
        self.P = Variable(torch.randn(3,4), requires_grad=True)
        self.default_mu = 5.0
        self.mus = nn.ModuleList()
        for i in range(K):
            w = nn.Linear(1,1, bias = False)
            w.weight = nn.Parameter(torch.FloatTensor([[self.default_mu]]),requires_grad = True)
            self.mus.append(w)
        '''
            
        self.noGradIdxs = []
        self.P = nn.Linear(K,K, bias = False)
        #self.P.weight = nn.Parameter(torch.zeros(K,K),requires_grad = True)
            
        self.pi = nn.Linear(K,1, bias = False)
        self.K = K

        self.NLL_weight = 0.5
        self.pi_weight  =0.5

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
        
    
    def form_P(self, inputLambda):
        self.noGradIdxs = [] 
        for i in range(self.K):
            for j in range(self.K):
                if i == 0 and j == 0:
                    #self.P.weight[i][j] = nn.Parameter(torch.FloatTensor([-inputLambda]),requires_grad = False)
                    self.P.weight[i][j].data.clamp_(min=-inputLambda.item(), max = -inputLambda.item())
                    self.noGradIdxs.append([i,j])
                elif i == j-1:
                    #self.P.weight[i][j] = nn.Parameter(torch.FloatTensor([inputLambda]),requires_grad = False)
                    self.P.weight[i][j].data.clamp_(min=inputLambda.item(), max = inputLambda.item())
                    self.noGradIdxs.append([i,j])
                elif i-1 == j:
                    pass
                elif i == j:
                    if i == self.K - 1 and j == self.K - 1:
                        #self.P.weight[i][j] =  -1*(self.P.weight[i][j])
                        pass
                    else:
                        #pass
                        #self.P.weight[i][j] =  self.P.weight[i][j] + inputLambda
                        tmp = self.P.weight[i][j].data.item() + inputLambda.item()
                        self.P.weight[i][j].data.clamp_(min=tmp, max =tmp)
                        #self.P.weight[i][j] =  -1*(self.P.weight[i][j] + inputLambda)
                        #self.P.weight[i][j] =  nn.Parameter(torch.FloatTensor([-1*(self.P.weight[i][j]+ inputLambda)]),requires_grad = True)
                else:
                    #self.P.weight[i][j] = nn.Parameter(torch.FloatTensor([0.0]),requires_grad = False)
                    self.P.weight[i][j].data.clamp_(min=0.0, max = 0.0)
                    self.noGradIdxs.append([i,j])
                
                        
    
    def zero_out_gradients_for_nonParameters(self):
        for idx in self.noGradIdxs:
            self.P.weight.grad[idx[0]][idx[1]].fill_(0)   
                
        

    def toStr(self):
        s = 'asd'
        return s

    def paramClip(self):
        pass

    def forwardPass(self, inp, clampNumbers=False):
        self.form_P(inp)
        probs = F.log_softmax(self.pi.weight)
        return probs[-1][-1]
        

    def predict(self, inp, clampNumbers):
        logPK = self.forwardPass(torch.FloatTensor([inp]), clampNumbers)
        if clampNumbers:
            PK = torch.exp(logPK)
            PK = torch.clamp(PK, min=0.0, max=1.0)
        else:
            PK = torch.exp(logPK)

        return PK.item()


    def objective(self, logPK, inputRates, actualDrops):
        # NLL = - [ (#dropped calls)*logPK + (#not dropped packets)*(1-logPK) ]

        NLL = -torch.sum(actualDrops * logPK + (inputRates - actualDrops) * torch.log(1 - torch.exp(logPK)))
        NLL = NLL / len(inputRates)

        steady_state_loss = F.mse_loss(torch.mm(F.softmax(self.pi.weight), self.P.weight), F.softmax(self.pi.weight))

        obj = self.NLL_weight * NLL + self.pi_weight * steady_state_loss

        return obj


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
        simulationFile = dir + fname  # sipp_data_UFF_Perdue_01_1_reduced_1.csv     UFF_Perdue_01_12_reduced

        curr_df = pd.read_csv(simulationFile, usecols=inputPacketsCols + droppedPacketsCols)
        curr_df.fillna(0, inplace=True)  # replace missing values (NaN) to zero9
        for j, curr_row in curr_df.iterrows():
            try:
                the_lambda = float(curr_row['CallRate(P)'])
                failed = float(curr_row['FailedCall(P)'])
                if failed > the_lambda or the_lambda <= 0:
                    continue
                # PK = failed/the_lambda
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
    # PK = torch.exp(logPK)
    # PK = torch.clamp(PK, min=0, max=1.0)
    loss = bernoulli_NLL(logPK, data_X, data_Y)
    totalLoss = loss.item()
    return totalLoss





def train(model, train_X, train_Y, valid_X, valid_Y, test_X, test_Y, epochs, batchsize, annelingAt=20,
          shuffleData=False, showBatches=False):
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

        # shuffle training data
        if shuffleData:
            rows = len(train_X)
            idxs = list(range(0, rows))
            random.shuffle(idxs)
            idxs = torch.LongTensor(idxs)
            train_X = train_X[idxs]
            train_Y = train_Y[idxs]

        for b in range(numOfBatchs):
            if b == 167:
                dbg = 1
            batch_X = train_X[b:b + batchsize]
            batch_Y = train_Y[b:b + batchsize]
            logPK = model.forwardPass(batch_X, clampNumbers=False)
            # PK = torch.exp(logPK)
            # PK = torch.clamp(PK, min=0, max = 1.0)

            loss = model.objective(logPK, batch_X, batch_Y)
            loss.backward()
            model.zero_out_gradients_for_nonParameters()
            totalLoss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), model.gradientClip, norm_type=model.gradientClipType)
            model.optimiser.step()
            model.optimiser.zero_grad()
            model.paramClip()

            if showBatches:
                print 'Batch=', b, ' Loss = ', loss.item(), 'lr=', model.lr
                print 'P=\n', model.P.weight
                print 'pi=\n', model.pi.weight
                print 'softmax(pi)=\n', F.softmax(model.pi.weight)
                # print 'm=',model.m.grad.item(), 'K=',model.K.grad.item(), 'mu=',model.mu.grad.item()
                print

        # trainLoss = evaluate(model, train_X, train_Y)
        validLoss = evaluate(model, valid_X, valid_Y) / float(len(valid_X))
        if len(test_X) != 0:
            testLoss = evaluate(model, test_X, test_Y) / float(len(test_X))
        else:
            testLoss = -1

        trainLoss = totalLoss / float(len(train_X))
        print 'Epoch', e, '--------------------------------------'
        print 'Total Train MSE loss = ', trainLoss, 'lr=', model.lr
        print 'Total Valid MSE loss = ', validLoss
        print 'Total Test MSE loss  = ', testLoss
        print 'm=', model.m.item(), 'K=', model.K.item(), 'mu=', model.mu.item()
        print '--------------------------------------'

        # if validLoss > prevValidLoss:
        if model.optim == 'SGD' and e % annelingAt == 0 and e != 0:
            prevValidLoss = validLoss
            model.lr /= 2.0
            for param_group in model.optimiser.param_groups:
                param_group['lr'] = model.lr

        prevValidLoss = validLoss

        if (validLoss < bestValidLoss) or (validLoss == bestValidLoss and trainLoss < train_bestValidLoss):
            bestValidLoss = validLoss
            bestModel = copy.deepcopy(model)
            test_bestValidLoss = testLoss
            train_bestValidLoss = trainLoss
            # params = '_m=%0.3f_K=%0.3f_mu=%0.3f'%(bestModel.m.item(), bestModel.K.item(), bestModel.mu.item())
            torch.save(bestModel, 'model_model_' + model.toStr())

        print '\t', 'train loss=', train_bestValidLoss, 'best valid=', bestValidLoss, 'test loss=', test_bestValidLoss, ' m=', bestModel.m.item(), 'K=', bestModel.K.item(), 'mu=', bestModel.mu.item()

    torch.save(bestModel, 'birthDeath_model')


def main():
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_4_K_DEFT_SCALE_86_2018.10.29-22.19.41/sipp_results/'
    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_CORES_2_K_100000_SCALE_43_2018.11.03-13.38.21/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.4+.5_CORE_1_K_425982_SCALE_24_2018.11.23-18.47.32/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_INVITE_CORE_1_K_425982_SCALE_17_2018.11.21-02.49.02/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_MSG_CORE_1_K_425982_SCALE_80_2018.11.21-04.56.39/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/results_SUBSCRIBE_CORE_1_K_425982_SCALE_36_2018.11.21-07.04.10/sipp_results/'

    # dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/results_ALL_.5+0_CORE_1_K_425982_SCALE_30_2018.11.23-20.55.11/sipp_results/'

    summaryFile = 'summary_data_dump.csv'

    random.seed(1111)
    trainQuota = 0.85
    validQuota = 0.15

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

    model = useGenericQueue_learn_MMmK(K=5, lr=0.01, gradientClip=0.25, gradientClipType=2, optim='Adam')

    train(model,
          train_X, train_Y,
          valid_X, valid_Y,
          test_X, test_Y,
          epochs=10000, batchsize=1, annelingAt=500,
          shuffleData=True,
          showBatches=True
          )


if __name__ == "__main__":
    main()
    print('DONE!')