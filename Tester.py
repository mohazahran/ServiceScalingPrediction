'''
Created on Oct 31, 2018

@author: mohame11
'''
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
#from learn_MMmK import *
#from simple_NN import *
#from sliding_NN import *
#from LSTM import *
import numpy as np
#from learn_MMmK_slidingWin import *
#from MMmK_LSTM import *
#from learn_MMmK_bernoulli import *
#from learn_genericQueue_fromData import *
#from useGenericQueue_learn_MMmK_multipleMus import *

#from Learning_Generic_Queue_MMmK import *
#from Learning_Generic_Queue import *
from Learning_Generic_Queue_final import *
#from Learning_Generic_Queue_final_customGradients import *
#from Learning_Generic_Queue_final_customGradients_autoDiff import *
from Learning_Generic_Queue_final_customGradients_manualDerivatives import *

class Tester(object):
    '''
    classdocs
    '''


    def __init__(self, model = None, direct = '', fname = '', scalingThreshold = 0.005, calls_to_packets = 1.0, interval = 1):
        self.model = model
        self.dir = direct
        self.fname = fname
        self.scalingThreshold = scalingThreshold
        self.calls_to_packets = calls_to_packets
        self.interval = interval
        
    
    
    def parseDataFile(self, fpath, inputPacketsCols, droppedPacketsCols):
        df = pd.read_csv(fpath, usecols = inputPacketsCols+droppedPacketsCols)
        df.fillna(0, inplace=True) # replace missing values (NaN) to zero
        return df
    
    
    def plot_est_empPK_batch(self):
        #using client data
        fpath = self.dir + self.fname
        inputPacketsCols = ['CallRate(P)']
        droppedPacketsCols = ['FailedCall(P)']
        interval = self.interval
        calls_to_packets = self.calls_to_packets
        scalingThreshold = self.scalingThreshold
        
        df = self.parseDataFile(fpath, inputPacketsCols, droppedPacketsCols)
        
        
        totalInputPackets = 0
        totalDroppedPackets = 0
        MSE = 0
        intervalCount = 0
        print 'time_interval empirical_PK estimated_PK squareLoss'
        est_PKs = []
        emp_PKs = []
        my_lambdas = []
        timeIntervals = []
        squaredLoss = 0
        skipFlag = False
        for i, row in df.iterrows():
            
            intervalCount += 1
            totalInputPackets = 0
            totalDroppedPackets = 0
          
            for inpCol in inputPacketsCols:
                totalInputPackets += float(row[inpCol]) * calls_to_packets
                
            for dropCol in droppedPacketsCols:
                totalDroppedPackets += float(row[dropCol]) * calls_to_packets
                
            mylmabda = totalInputPackets
            try:
                empirical_PK = float(totalDroppedPackets) / totalInputPackets
                
            except:
                skipFlag = True
                
            if (totalDroppedPackets == 0 and totalInputPackets == 0) or totalInputPackets == 0 or empirical_PK > 1 or skipFlag:
                skipFlag = False
                continue
            #my_lambdas.append([mylmabda])
            my_lambdas.append(mylmabda)
            emp_PKs.append(empirical_PK)
            timeIntervals.append(intervalCount)
            
            
        my_lambdas.append(-1)
        #emp_PKs.append(-1)
        batch_X, batch_Y = self.model.format_trainingData(my_lambdas, emp_PKs)
        batch_X = batch_X[0]
        batch_Y = batch_Y[0]
        pred = self.model.predict(batch_X)
        pred = torch.clamp(pred, min = 0, max = 1)
        est_PKs = list(pred.data.numpy())
        
        
        fig = plt.figure(1, figsize=(6, 4))
        #plt.xticks([x for x in range(1,max(timeIntervals))])
        axes = plt.gca()
        ax = plt.axes()
        #axes.set_ylim([-0.05,0.6])
        
        
        #drawing est PK vs. emp PK
        plt.ylabel('Probability')
        plt.xlabel('Time Interval(1 interval=' +str(interval)+' s)')
        lines = plt.plot(timeIntervals, est_PKs, '--r' ,label='Estimated PK') 
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(timeIntervals, emp_PKs, 'b' ,label='Empirical PK') 
        plt.setp(lines, linewidth=2.0)
        
        #finding the time for scaling
        scalingTime = -1
        criticalPK = 0
        for i in range(len(est_PKs)):
            if est_PKs[i] > scalingThreshold: 
                scalingTime = timeIntervals[i]
                criticalPK = est_PKs[i]
                break
        
        max_y = max(max(est_PKs), max(emp_PKs))
        y = list(np.arange(0.0, max_y, max_y/5))
        ax.text(scalingTime, max(y), 't='+str(scalingTime)+'\nProb Thresh.='+str(scalingThreshold), fontsize=12)
        lines = plt.plot([scalingTime for t in y], np.arange(0.0, max_y, max_y/5), ':g' ,label='Scaling Time') 
        plt.setp(lines, linewidth=2.0)
        
        plt.legend(loc = 2, prop={'size':17}, labelspacing=0.1) 
        fig.suptitle(self.fname, fontsize=12, fontweight='bold', horizontalalignment='center', y=.86)
        plt.grid()                                                                     
        #plt.savefig(resultsPath+'combined_rec_prec_plot_withActionSampling.pdf', bbox_inches='tight')
        plt.show()
        
        
    def plot_est_empPK_batch_mixTraffic(self, models, inputTrafficWeights, outputTrafficWeights):
        #using client data
        fpath = self.dir + self.fname
        inputPacketsCols = ['CallRate(P)']
        droppedPacketsCols = ['FailedCall(P)']
        interval = self.interval
        calls_to_packets = self.calls_to_packets
        scalingThreshold = self.scalingThreshold
        
        df = self.parseDataFile(fpath, inputPacketsCols, droppedPacketsCols)
        
        
        totalInputPackets = 0
        totalDroppedPackets = 0
        MSE = 0
        intervalCount = 0
        print 'time_interval empirical_PK estimated_PK squareLoss'
        est_PKs = []
        emp_PKs = []
        my_lambdas = []
        timeIntervals = []
        squaredLoss = 0
        skipFlag = False
        for i, row in df.iterrows():
            
            intervalCount += 1
            totalInputPackets = 0
            totalDroppedPackets = 0
          
            for inpCol in inputPacketsCols:
                totalInputPackets += float(row[inpCol]) * calls_to_packets
                
            for dropCol in droppedPacketsCols:
                totalDroppedPackets += float(row[dropCol]) * calls_to_packets
                
            mylmabda = totalInputPackets
            try:
                empirical_PK = float(totalDroppedPackets) / totalInputPackets
                
            except:
                skipFlag = True
                
            if (totalDroppedPackets == 0 and totalInputPackets == 0) or totalInputPackets == 0 or empirical_PK > 1 or skipFlag:
                skipFlag = False
                continue
            #my_lambdas.append([mylmabda])
            my_lambdas.append(mylmabda)
            emp_PKs.append(empirical_PK)
            timeIntervals.append(intervalCount)
            
        
        
        for i in range(len(my_lambdas)):
            totalPred = 0.0
            for typ in models:
                pred = models[typ].predict(my_lambdas[i]*inputTrafficWeights[typ], clampNumbers = True)
                pred *= outputTrafficWeights[typ]
                totalPred += pred
            est_PKs.append(totalPred)
            
        
        
        fig = plt.figure(1, figsize=(6, 4))
        #plt.xticks([x for x in range(1,max(timeIntervals))])
        axes = plt.gca()
        ax = plt.axes()
        #axes.set_ylim([-0.05,0.6])
        
        
        #drawing est PK vs. emp PK
        plt.ylabel('Probability')
        plt.xlabel('Time Interval(1 interval=' +str(interval)+' s)')
        lines = plt.plot(timeIntervals, est_PKs, '--r' ,label='Estimated PK') 
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(timeIntervals, emp_PKs, 'b' ,label='Empirical PK') 
        plt.setp(lines, linewidth=2.0)
        
        
        
        plt.legend(loc = 2, prop={'size':17}, labelspacing=0.1) 
        fig.suptitle(self.fname, fontsize=12, fontweight='bold', horizontalalignment='center', y=.86)
        plt.grid()                                                                     
        #plt.savefig(resultsPath+'combined_rec_prec_plot_withActionSampling.pdf', bbox_inches='tight')
        plt.show()  
        
    
    def plot_estPK_empPK(self):
        #using client data
        fpath = self.dir + self.fname
        inputPacketsCols = ['CallRate(P)']
        droppedPacketsCols = ['FailedCall(P)']
        interval = self.interval
        calls_to_packets = self.calls_to_packets
        scalingThreshold = self.scalingThreshold
        
        df = self.parseDataFile(fpath, inputPacketsCols, droppedPacketsCols)
        
        if interval == -1:
            interval = df.shape[0]-1
        
        totalInputPackets = 0
        totalDroppedPackets = 0
        MSE = 0
        NLL = 0
        intervalCount = 0
        print 'intervalCount my_lambda, empirical_PK, estimated_PK, realDrops, estDrops'
        est_PKs = []
        emp_PKs = []
        my_lambdas = []
        drops = []
        timeIntervals = []
        squaredLoss = 0
        skipFlag = False
        for i, row in df.iterrows():
            if i % interval == 0 and i != 0:
                intervalCount += 1
                if intervalCount == 134:
                    dbg = 1
                
                avgDroppedPackets = totalDroppedPackets / float(interval)
                avgInputPackets   = totalInputPackets / float(interval)
                
                my_lambda = avgInputPackets
                
                try:
                    empirical_PK = float(totalDroppedPackets) / totalInputPackets
                except:
                    skipFlag = True
                    
                if (totalDroppedPackets == 0 and totalInputPackets == 0) or totalInputPackets == 0 or empirical_PK > 1:
                    #empirical_PK = 0
                    skipFlag = True
                    
                
                try:
                    estimated_PK = 0.0
                    for m in self.model:
                        estimated_PK += m.predict(my_lambda, clampNumbers = True)
                    estimated_PK /= len(self.model)
                    
                except:
                    skipFlag = True
                    
                if skipFlag:
                    skipFlag = False
                    totalInputPackets = 0
                    totalDroppedPackets = 0
                   
                    for inpCol in inputPacketsCols:
                        totalInputPackets += row[inpCol] * calls_to_packets
                        
                    for dropCol in droppedPacketsCols:
                        totalDroppedPackets += row[dropCol] * calls_to_packets
                    
                    continue
                
                
                    
                     
                timeIntervals.append(intervalCount)
                emp_PKs.append(empirical_PK)
                est_PKs.append(estimated_PK)
                my_lambdas.append(my_lambda) 
                drops.append(avgDroppedPackets)
                
                squaredLoss = (empirical_PK - estimated_PK)**2
                MSE += squaredLoss

                
                print intervalCount, my_lambda, empirical_PK, estimated_PK, avgDroppedPackets, estimated_PK*my_lambda
                
                totalInputPackets = 0
                totalDroppedPackets = 0
               
                for inpCol in inputPacketsCols:
                    totalInputPackets += row[inpCol] * calls_to_packets
                    
                for dropCol in droppedPacketsCols:
                    totalDroppedPackets += row[dropCol] * calls_to_packets
               
                
            else:
                
                for inpCol in inputPacketsCols:
                    totalInputPackets += row[inpCol] * calls_to_packets
                    
                for dropCol in droppedPacketsCols:
                    totalDroppedPackets += row[dropCol] * calls_to_packets

        NLL = -torch.sum(torch.FloatTensor(drops) * torch.log(torch.FloatTensor(est_PKs)) + (torch.FloatTensor(my_lambdas) - torch.FloatTensor(drops)) * torch.log(1 - torch.FloatTensor(est_PKs)))
        NLL = NLL / float(len(my_lambdas))

        print 'MSE=', squaredLoss/intervalCount
        print 'NLL=', NLL.item()
        print '#drops=', sum(drops)
        
        fig = plt.figure(1, figsize=(6, 4))
        #plt.xticks([x for x in range(1,max(timeIntervals))])
        axes = plt.gca()
        ax = plt.axes()
        #axes.set_ylim([-0.05,0.6])
        
        
        #drawing est PK vs. emp PK
        plt.ylabel('Probability')
        plt.xlabel('Time Interval(1 interval=' +str(interval)+' s)')
        lines = plt.plot(timeIntervals, est_PKs, '--r' ,label='Estimated drop probability')
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(timeIntervals, emp_PKs, 'b' ,label='Empirical drop probability')
        plt.setp(lines, linewidth=2.0)
        
        '''
        #finding the time for scaling
        scalingTime = -1
        criticalPK = 0
        for i in range(len(est_PKs)):
            if est_PKs[i] > scalingThreshold: 
                scalingTime = timeIntervals[i]
                criticalPK = est_PKs[i]
                break
        
        max_y = max(max(est_PKs), max(emp_PKs))
        y = list(np.arange(0.0, max_y, max_y/5))
        ax.text(scalingTime, max(y), 't='+str(scalingTime)+'\nProb Thresh.='+str(scalingThreshold), fontsize=12)
        lines = plt.plot([scalingTime for t in y], np.arange(0.0, max_y, max_y/5), ':g' ,label='Scaling Time') 
        plt.setp(lines, linewidth=2.0)
        '''
        
        plt.legend(loc = 2, prop={'size':17}, labelspacing=0.1) 
        #fig.suptitle(self.fname, fontsize=12, fontweight='bold', horizontalalignment='center', y=.86)
        plt.grid()                                                                     
        #plt.savefig(resultsPath+'combined_rec_prec_plot_withActionSampling.pdf', bbox_inches='tight')
        plt.show()



        plt.ylabel('#drops')
        plt.xlabel('Time Interval(1 interval=' + str(interval) + ' s)')
        lines = plt.plot(timeIntervals, [est_PKs[i]*my_lambdas[i] for i in range(len(est_PKs))], '--r', label='Estimated #drops')
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(timeIntervals, drops, 'b', label='Real #drops')
        plt.setp(lines, linewidth=2.0)
        plt.legend(loc=2, prop={'size': 17}, labelspacing=0.1)
        plt.grid()
        plt.show()
        
        
        #showing the input rate
        fig2 = plt.figure(1, figsize=(6, 4))
        #plt.xticks([x for x in range(1,max(timeIntervals))])
        axes = plt.gca()
        ax = plt.axes()
        plt.ylabel('Calls/s')
        plt.xlabel('Time Interval(1 interval=' +str(interval)+' s)')
        lines = plt.plot(timeIntervals, my_lambdas, 'b' ,label='Input rate') 
        plt.setp(lines, linewidth=2.0)
        lines = plt.plot(timeIntervals, drops, '--g' ,label='Drop rate') 
        plt.setp(lines, linewidth=2.0)
        plt.legend(loc = 2, prop={'size':17}, labelspacing=0.1) 
        fig2.suptitle(self.fname, fontsize=12, fontweight='bold', horizontalalignment='center', y=.86)
        plt.grid()                                                                     
        plt.show()
        
    
    
        
        
        
    def getData(self, direct = '', summaryFile = 'summary_data_dump.csv', minDropRate = 0, maxDropRate = 1e100):
        sfile = direct + summaryFile
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
            simulationFile = direct + fname #sipp_data_UFF_Perdue_01_1_reduced_1.csv     UFF_Perdue_01_12_reduced
            
            curr_df = pd.read_csv(simulationFile, usecols = inputPacketsCols+droppedPacketsCols)
            curr_df.fillna(0, inplace=True) # replace missing values (NaN) to zero
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
                
        return train_X, train_Y
    
    def cal_MSE(self, data_X, data_Y):
        squaredLoss = 0.0
        for i in range(len(data_X)):
            pred = self.model[0].predict(data_X[i], clampNumbers = True)
            squaredLoss += (pred - data_Y[i])**2
        
        squaredLoss /= len(data_X)
        return squaredLoss
            
        
def ensemble():
    modelNames = {'invite' :'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_invite_model_L',
                  'msg': 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_msg_model_O',
                  'subsc': 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_subsc_model_P'
                  }
    
    models = { nm : torch.load(modelNames[nm]) for nm in modelNames}
    
    direct = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/'
    
    #direct += '/results_ALL_.4+.5_CORE_1_K_425982_SCALE_24_2018.11.23-18.47.32/sipp_results/'
    #inputTrafficWeights = {'invite':0.7618, 'msg':0.1547, 'subsc': 0.1586}
    #outputTrafficWeights = {'invite':0.6739, 'msg':0.1279, 'subsc': 0.1034}
    
    #inputTrafficWeights = {'invite':0.4075, 'msg':0.3893, 'subsc': 0.9792}
    #outputTrafficWeights = {'invite':0.2405, 'msg':0.1756, 'subsc': 0.7661}
    
    direct += '/results_ALL_.5+0_CORE_1_K_425982_SCALE_30_2018.11.23-20.55.11/sipp_results/'
    #trafficShares = {'invite':0.5, 'msg':0.5, 'subsc': 0.0}
    #inputTrafficWeights = {'invite':0.5, 'msg':0, 'subsc': 0.5}
    #outputTrafficWeights = {'invite':1, 'msg':1, 'subsc': 1}
    
    inputTrafficWeights = {'invite':0.4448, 'msg':0.4044, 'subsc': 1.0759}
    outputTrafficWeights = {'invite':0.2779, 'msg':0.3486, 'subsc': 0.8251}
    
    
    #fname = 'sipp_raw_data_UFF_Perdue_07_7_reduced_1.csv'
    fname = 'sipp_raw_data_UFF_Perdue_04_42_reduced_1.csv'
    
    t = Tester(model = None, direct = direct, fname = fname, scalingThreshold = 0.005, calls_to_packets = 1.0, interval = 1)
    
    t.plot_est_empPK_batch_mixTraffic(models, inputTrafficWeights, outputTrafficWeights)
    
    
    
    '''
    data_X, data_Y = t.getData(direct = direct, summaryFile='summary_data_dump.csv', minDropRate = -1, maxDropRate = 1e100)
    squaredLoss = 0.0
    for i in range(len(data_X)):
        totalPred = 0.0
        #totalPred = []
        for typ in models:
            
            pred = models[typ].predict(data_X[i]*inputTrafficWeights[typ], clampNumbers = True)
            pred *= outputTrafficWeights[typ]
            totalPred += pred
            #totalPred.append(pred)
        #totalPred /= len(models)
        squaredLoss += (totalPred - data_Y[i])**2
        #squaredLoss += (max(totalPred) - data_Y[i])**2
    
    squaredLoss /= len(data_X)
    print '#test samples=',len(data_X)
    print 'MSE=',squaredLoss
    '''



def main():
    mpl.rcParams.update({'font.size': 17})
    #modelName = 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_restricted'
    #modelName = 'MMmK_model_slidingm0=1.0_K0=5.0_mu0=5.0_4cores'
    #modelName = 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_fullData'
    #modelName = 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_r_fullData'
    #modelName = 'MMmK_LSTMm0=1.0_K0=5.0_mu0=5.0_4cores'
    #modelName = 'LSTM_model'
    #modelName = 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_r_mixed_'
    #modelName = 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_invite_model_L'
    #modelName = 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_msg_model_O'
    #modelName = 'MMmK_model_m0=1.0_K0=5.0_mu0=5.0_mixed_.5_0_.5_model_R'

    #modelName = 'generic_MMmK2'
    #modelName = 'generic_MMmK_multiple_mus'
    #modelName = 'generic_MMmK2_learningSteadyState'
    #modelName = 'MMmK_model_asd'
    #modelName = 'MMmK_model_asd'

    #modelName = 'MMmK_model_bernoullim0=5.0_K0=5.0_mu0=5.0'

    #modelName = 'genericQueueModel_multipleMus_K5_m5_pi'
    modelName = 'realData2_muPerState_lowDrops2500_K5'


    model = torch.load(modelName)
    #print model.params
    print model.mus_2_str()

    #model.mu.data.clamp_(min = 660.3567, max = 660.3567)
    
    direct = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/'
    #direct += '/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    direct += '/results_INVITE_CORE_1_K_425982_SCALE_60_REMOTE_CPU_2019.01.08-01.56.08/sipp_results/'
    #direct += '/results_CORES_2_K_DEFT_SCALE_43_2018.10.29-18.56.45/sipp_results/'
    #direct += '/results_CORES_3_K_DEFT_SCALE_64_2018.10.29-20.38.11/sipp_results/'
    #direct += '/results_CORES_4_K_DEFT_SCALE_86_2018.10.29-22.19.41/sipp_results/'
    
    
    
    #direct = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/'
    #direct += '/results_ALL_.4+.5_CORE_1_K_425982_SCALE_24_2018.11.23-18.47.32/sipp_results/'
    
    #direct = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/Mixed/'
    #direct += '/results_ALL_.5+0_CORE_1_K_425982_SCALE_30_2018.11.23-20.55.11/sipp_results/'
    
    
    
    #direct = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/'
    #direct += 'results_INVITE_CORE_1_K_425982_SCALE_17_2018.11.21-02.49.02/sipp_results/'
    
    #direct = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/'
    #direct += 'results_MSG_CORE_1_K_425982_SCALE_80_2018.11.21-04.56.39/sipp_results/'
    
    #direct = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Traffic-Types/'
    #direct += 'results_SUBSCRIBE_CORE_1_K_425982_SCALE_36_2018.11.21-07.04.10/sipp_results/'
    
    
    #fname = 'sipp_raw_data_UFF_Perdue_07_7_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_02_10_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_01_27_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_08_14_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_01_13_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_04_23_reduced_1.csv'
    #fname = 'sipp_raw_data_UFF_Perdue_02_29_reduced_1.csv'

    #fname = 'sipp_raw_data_UFF_Perdue_04_42_reduced_1.csv'

    fname = 'sipp_data_long_var_rate_0_1836_seconds_1.csv'
    
    
    
    t = Tester(model = [model], direct = direct, fname = fname, scalingThreshold = 0.005, calls_to_packets = 1.0, interval = 1)
    
    t.plot_estPK_empPK()
    #t.plot_est_empPK_batch()
    
    '''
    data_X, data_Y = t.getData(direct = direct, summaryFile='summary_data_dump.csv', minDropRate = -1, maxDropRate = 1e100)
    MSE = t.cal_MSE(data_X, data_Y)
    print 'dir=', direct
    print 'Model Name = ', modelName
    print '#test samples=',len(data_X)
    print 'MSE=', MSE
    '''
    

if __name__ == "__main__":
    #ensemble()
    main()
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    