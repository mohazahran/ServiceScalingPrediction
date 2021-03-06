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



class GenericQueue(nn.Module):
    def __init__(self, params):
        super(GenericQueue, self).__init__()
        
        self.params = params
        
        self.mu = nn.Parameter(torch.FloatTensor([params['initialMu']]*(self.params['K']-1)),requires_grad = True)
        self.Q = torch.zeros(self.params['K'], self.params['K'])
        self.optimiser = getattr(torch.optim, params['optim'])(self.parameters(), lr = params['lr'])
        
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
        if self.params['cuda'] == True:
            self.Q = torch.zeros((self.params['K'], self.params['K'])).cuda()
        else:
            self.Q = torch.zeros((self.params['K'], self.params['K']))
        for i in range(self.params['K']):
            for j in range(self.params['K']):
                if i == 0 and j == 0:
                    self.Q[i][j] = -inputLambda
                elif i == j-1:
                    self.Q[i][j] = inputLambda
                elif i-1 == j:
                    self.Q[i][j] = self.mu[i-1]
                elif i == j:
                    if i == self.params['K'] - 1 and j == self.params['K'] - 1:
                        self.Q[i][j] = -1*self.mu[i-1]

                    else:
                        self.Q[i][j] = -1*(self.mu[i-1] + inputLambda)
                else:
                    self.Q[i][j] = 0.0
        return self.Q

    def zero_out_gradients_for_nonParameters(self):
        if self.Q.grad:
            for idx in self.noGradIdxs:
                self.Q.grad[idx[0]][idx[1]].fill_(0)

    def toStr(self):
        s = 'asd'
        return s

    def paramClip(self):
        self.mu.data.clamp_(min=1.0, max=1e10)



    def objective_batch(self, inputLa, dropCount, dropProb):
        if self.params['cuda'] == True:
            bigP = torch.zeros((len(inputLa), self.params['K'], self.params['K'])).cuda()
        else:
            bigP = torch.zeros((len(inputLa), self.params['K'], self.params['K']))
            
        for i in range(len(inputLa)):
            Q = self.form_Q(inputLa[i])
            if self.params['cuda'] == True:
                I = torch.diag(torch.ones(self.params['K'])).cuda()
                g = torch.max ( I * torch.abs(Q) )
                P = torch.diag(torch.ones(self.params['K'])).cuda() + torch.div(Q, g).cuda()
                bigP[i] = P
            else:
                I = torch.diag(torch.ones(self.params['K']))
                g = torch.max ( I * torch.abs(Q) )
                P = torch.diag(torch.ones(self.params['K'])) + torch.div(Q, g)
                bigP[i] = P
            
        
        
        for t in range(self.params['t']):
            bigP = torch.matmul(bigP, bigP)
            steady_state_loss = 0

        if self.params['calculatePK'] == 'AVG':
            #PK = torch.mean(Pt[:,-1])
            PK = torch.mean(bigP[:,:,-1], dim=1)
        elif self.params['calculatePK'] == 'MIN':
            #PK = torch.min(Pt[:,-1])
            PK = torch.min(bigP[:,:,-1], dim=1)
        elif self.params['calculatePK'] == 'MAX':
            #PK = torch.max(Pt[:,-1])
            PK = torch.max(bigP[:,:,-1], dim=1)


        logPK = torch.log(PK)
        if self.params['dataLossType'] == 'MSE':
            dataLoss = F.mse_loss(PK, torch.FloatTensor([dropProb]))
        elif self.params['dataLossType'] == 'NLL':
            dataLoss = -1 * (dropCount * logPK + (inputLa - dropCount) * torch.log(1 - torch.exp(logPK)))
        dataLoss = torch.sum(dataLoss) / float(len(PK))

        loss = self.params['dataLossWeight'] * dataLoss + self.params['steadyStateWeight'] * steady_state_loss

        return loss, logPK



    def objective(self, inputLa, dropCount, dropProb):
        self.form_Q(inputLa)
        if self.params['cuda'] == True:
            I = torch.diag(torch.ones(self.params['K'])).cuda()
            g = torch.max ( I * torch.abs(self.Q) )
            P = torch.diag(torch.ones(self.params['K'])).cuda() + torch.div(self.Q, g).cuda()
        else:
            I = torch.diag(torch.ones(self.params['K']))
            g = torch.max ( I * torch.abs(self.Q) )
            P = torch.diag(torch.ones(self.params['K'])) + torch.div(self.Q, g)
        cnt = 1
        Pt_1 = torch.mm(P,P)
        while True:
            cnt += 1
            Pt = torch.mm(Pt_1, Pt_1)
            #print Pt_1.shape, Pt_1[-1].shape, Pt_1[-1].view(1, Pt_1.shape[0]).shape
            Pt_1_K = Pt_1[-1].view(1, Pt_1.shape[0])
            if self.params['steadyStateLossType'] == 'L2':
                steady_state_loss = F.mse_loss(torch.mm(Pt_1_K, Pt), Pt_1_K)
            elif self.params['steadyStateLossType'] == 'L1':
                steady_state_loss = torch.sum( torch.abs( torch.mm(Pt_1_K, P) - Pt_1_K) )

            if cnt > self.params['t'] or steady_state_loss < self.params['steadyStateEpsilon']:
                break

            Pt_1 = Pt


        if self.params['calculatePK'] == 'AVG':
            PK = torch.mean(Pt[:,-1])
        elif self.params['calculatePK'] == 'MIN':
            PK = torch.min(Pt[:,-1])
        elif self.params['calculatePK'] == 'MAX':
            PK = torch.max(Pt[:,-1])


        logPK = torch.log(PK)
        if self.params['dataLossType'] == 'MSE':
            dataLoss = F.mse_loss(PK, torch.FloatTensor([dropProb]))
        elif self.params['dataLossType'] == 'NLL':
            dataLoss = -1 * (dropCount * logPK + (inputLa - dropCount) * torch.log(1 - torch.exp(logPK)))

        loss = self.params['dataLossWeight'] * dataLoss + self.params['steadyStateWeight'] * steady_state_loss

        return loss, logPK

    def predict(self, la, clampNumbers = False):
        self.form_Q(la)
        if self.params['cuda'] == True:
            I = torch.diag(torch.ones(self.params['K'])).cuda()
            g = torch.max ( I * torch.abs(self.Q) )
            P = torch.diag(torch.ones(self.params['K'])).cuda() + torch.div(self.Q, g)
        else:
            I = torch.diag(torch.ones(self.params['K']))
            g = torch.max ( I * torch.abs(self.Q) )
            P = torch.diag(torch.ones(self.params['K'])) + torch.div(self.Q, g)
        cnt = 1
        Pt_1 = torch.mm(P,P)
        while True:
            Pt = torch.mm(Pt_1, Pt_1)
            cnt += 1
            #print Pt_1.shape, Pt_1[-1].shape, Pt_1[-1].view(1, Pt_1.shape[0]).shape
            Pt_1_K = Pt_1[-1].view(1, Pt_1.shape[0])
            if self.params['steadyStateLossType'] == 'L2':
                steady_state_loss = F.mse_loss(torch.mm(Pt_1_K, Pt), Pt_1_K)
            elif self.params['steadyStateLossType'] == 'L1':
                steady_state_loss = torch.sum( torch.abs( torch.mm(Pt_1_K, P) - Pt_1_K) )

            if cnt > self.params['t'] or steady_state_loss < self.params['steadyStateEpsilon']:
                break

            Pt_1 = Pt

        if self.params['calculatePK'] == 'AVG':
            PK = torch.mean(Pt[:,-1])
        elif self.params['calculatePK'] == 'MIN':
            PK = torch.min(Pt[:,-1])
        elif self.params['calculatePK'] == 'MAX':
            PK = torch.max(Pt[:,-1])

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
                PK = failed/the_lambda
                #PK = failed
            except:
                continue
            train_X.append(Variable(torch.FloatTensor([the_lambda])))
            train_Y.append(Variable(torch.FloatTensor([PK])))

    return train_X, train_Y




def evaluate_given_model(testModel, testLambdas, testPKs, plotFig = True):
    testModel.eval()
    est_PKs = []
    avg_NLL = 0.0
    for i in range(len(testLambdas)):
        est_PK = testModel.predict(testLambdas[i])
        logPK = math.log(est_PK)
        est_PKs.append(est_PK)
        actualDrops = testPKs[i] * testLambdas[i]
        NLL = - (actualDrops * logPK + (testLambdas[i] - actualDrops) * math.log(1 - math.exp(logPK)))
        avg_NLL += NLL

    if testModel.params['cuda'] == True:
        est_PK_torch = torch.FloatTensor(est_PKs).cuda()
        testPKs_torch = torch.FloatTensor(testPKs).cuda()
    else:
        est_PK_torch = torch.FloatTensor(est_PKs)
        testPKs_torch = torch.FloatTensor(testPKs)

    MSE = F.mse_loss(est_PK_torch, testPKs_torch)
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

    testModel.train()
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
    bestValidLoss = 1e10
    bestModel = model

    if model.params['cuda'] == True:
        inputLambdas = Variable(torch.FloatTensor(inputLambdas)).cuda()
        PKs = Variable(torch.FloatTensor(PKs)).cuda()

        validLambdas = Variable(torch.FloatTensor(validLambdas)).cuda()
        validPKs = Variable(torch.FloatTensor(validPKs)).cuda()
    else:
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

        if model.params['batchify'] == False:
            for b in range(len(inputLambdas)):

                inputLa = inputLambdas[b]
                dropProb = PKs[b]
                dropCount = dropProb * inputLa

                loss, logPK = model.objective(inputLa, dropCount, dropProb)

                est_PKs.append(torch.exp(logPK).item())
                true_PKs.append(PKs[b])

                total_loss = total_loss + loss
                epochLoss += loss.item()

                if (b+1) % batchSize == 0:
                    total_loss = total_loss / float(batchSize)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), model.params['gradientClip'], norm_type=model.params['gradientClipType'])
                    model.optimiser.step()
                    model.optimiser.zero_grad()
                    if showBatches:
                        print 'batch#',b ,'loss=',total_loss.item(), 'mu=', model.mu.data
                    total_loss = 0.0

        else: # batchify = True
            numOfBatches = len(inputLambdas) / batchSize
            for b in range(numOfBatches):
                inputLa = inputLambdas[b*batchSize : b*batchSize + batchSize]
                dropProb = PKs[b*batchSize : b*batchSize + batchSize]
                dropCount = inputLa * dropProb

                total_loss, logPK = model.objective_batch(inputLa, dropCount, dropProb)
                for j in range(len(inputLa)):
                    est_PKs.append(torch.exp(logPK[j]).item())
                    true_PKs.append(dropProb[j])
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), model.params['gradientClip'], norm_type=model.params['gradientClipType'])
                model.optimiser.step()
                model.optimiser.zero_grad()
                if showBatches:
                    print 'batch#',b ,'loss=',total_loss.item(), 'mu=', model.mu.data
                total_loss = 0.0

        total_loss = 0.0
        est_PKs = torch.FloatTensor(est_PKs)
        true_PKs = torch.FloatTensor(true_PKs)
        MSE = F.mse_loss(est_PKs, true_PKs)

        validMSE, validNLL = evaluate_given_model(model, validLambdas, validPKs, plotFig = False)
        if model.params['dataLossType'] == 'MSE':
            validLoss = validMSE
        elif model.params['dataLossType'] == 'NLL':
            validLoss = validNLL

        if validLoss < bestValidLoss:
            bestValidLoss = validLoss
            torch.save(model, modelName)
            bestModel = torch.load(modelName)

        print 'epoch=', e , 'lr=', model.params['lr'], 'mu=',model.mu.data
        print 'TrainLoss= ', epochLoss/float(len(inputLambdas))
        print 'MSE=', MSE.item()
        print 'validLoss=', validLoss
        print '\tbest validLoss=', bestValidLoss
        #print '\tbest mu=',bestModel.mu.data
        print

        torch.save(bestModel, modelName)







def run_using_MMmK_simulation():

    true_m = 3
    true_mu = 50.0
    true_K = 5

    maxLambda = 10000
    inputDataSize = 400
    #train_X = list(range(1,25,2))
    train_X = list(range(1, maxLambda, 10))
    random.shuffle(train_X)
    train_X = train_X[:inputDataSize]
    #train_X = [10]

    train_Y = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K) )  for inp in train_X ]

    #valid_X = list(range(2,25,7))
    valid_X = list(range(1, maxLambda, 7))
    random.shuffle(valid_X)
    valid_X = valid_X[:inputDataSize]
    valid_Y = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K) )  for inp in valid_X ]

    ########################################
    params = {
        'cuda' : False,
        'K':5,
        'modelName' : 'mimicing_MMmK_generic',
        'dataLossType' : 'NLL', #NLL or  MSE
        'steadyStateLossType' : 'L1', #L1 or L2
        't': min(5, 10),
        'steadyStateEpsilon' : 10e-6,
        'dataLossWeight' : 1.0,
        'steadyStateWeight' : 0.0,
        'calculatePK' : 'AVG', #MIN, MAX, AVG,
        'initialMu' : 25.0,
        'optim' : 'Adam',
        'lr' : 0.05,
        'gradientClip' : 10.0,
        'gradientClipType' : 2
    }

    print 'Parameters:\n', params

    if torch.cuda.is_available():
        print 'torch.cuda.is_available() = ', torch.cuda.is_available()


    random.seed(1111)
    if params['cuda'] == True:
        torch.cuda.manual_seed(1111)
    else:
        torch.manual_seed(1111)


    model = GenericQueue(params)

    if params['cuda'] == True:
        model.cuda()
    else:
        torch.set_num_threads(8)



    train(model = model,
          inputLambdas = train_X, PKs = train_Y,
          batchSize = 1,
          epochs = 50,
          validLambdas = valid_X, validPKs = valid_Y,
          modelName = params['modelName'],
          shuffleData = True,
          showBatches = False
          )



    #testLambdas = list(range(1,25,1))
    testLambdas = list(range(1, maxLambda, 1))
    testPKs = [ math.exp(solver.M_M_m_K_log(float(inp)/float(true_mu), true_m, true_K) )  for inp in testLambdas ]
    evaluate_given_model(model, testLambdas, testPKs)








def run_using_real_data():
    random.seed(1111)

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
    params = {
        'cuda' : False,
        'K': 8,
        't': min(8, 10),
        'batchify' : False,
        'modelName' : 'generic_MMmK_multiple_mus',
        'dataLossType' : 'NLL', #NLL or  MSE
        'steadyStateLossType' : 'L1', #L1 or L2
        'steadyStateEpsilon' : 10e-6,
        'dataLossWeight' : 1.0,
        'steadyStateWeight' : 0.0,
        'calculatePK' : 'AVG', #MIN, MAX, AVG,
        'initialMu' : 50.0,
        'optim' : 'Adam',
        'lr' : 0.05,
        'gradientClip' : 10.0,
        'gradientClipType' : 2
    }

    print 'Parameters:\n', params

    #print torch.__version__

    #print torch.version.cuda

    if torch.cuda.is_available():
        print 'torch.cuda.is_available() = ', torch.cuda.is_available()



    if params['cuda'] == True:
        torch.cuda.manual_seed(1111)
    else:
        torch.manual_seed(1111)


    model = GenericQueue(params)

    if params['cuda'] == True:
        model.cuda()
    else:
        torch.set_num_threads(8)



    train(model = model,
          inputLambdas = train_X, PKs = train_Y,
          batchSize = 32,
          epochs = 100,
          validLambdas = valid_X, validPKs = valid_Y,
          modelName = params['modelName'],
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
    #run_using_MMmK_simulation()
    run_using_real_data()
    print('DONE!')