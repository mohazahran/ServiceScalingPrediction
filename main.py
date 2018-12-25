'''
Created on Sep 28, 2018

@author: mohame11
'''
import pandas as pd
import solver
import math
import numpy as np
from scipy.optimize import *
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def parseDataFile(fpath, inputPacketsCols, droppedPacketsCols):
    df = pd.read_csv(fpath, usecols = inputPacketsCols+droppedPacketsCols)
    df.fillna(0, inplace=True) # replace missing values (NaN) to zero
    
    #df['inputPacketsCols'] = df[inputPacketsCols[0]] + df[inputPacketsCols[1]]
    #df['droppedPacketsCols'] = df[droppedPacketsCols[0]] + df[droppedPacketsCols[1]]
    
    return df
    


def PK_estimation_lambdaFromData(K=167, m=1, mu=1250, scalingThreshold = 0.005, calls_to_packets = 6.0):
    '''
    #const rate
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_1_2018.09.28-19.40.18/kamailio_results/'
    fname = 'kamailio_data_const_rate_1200_1800_10_50_50.csv'
    fpath = dir + fname
    inputPacketsCols = ['core:rcv_replies', 'core:rcv_requests']
    droppedPacketsCols = ['sl:404_replies', 'System:Packet_Drops']
    interval = 10
    '''
    
    '''
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/var_rate/kamailio_results/'
    fname = 'kamailio_data_UFF_Perdue_10_1_reduced.csv'
    fpath = dir + fname
    inputPacketsCols = ['core:rcv_replies', 'core:rcv_requests']
    droppedPacketsCols = ['sl:404_replies', 'System:Packet_Drops']
    interval = 1
    '''
    #using client data
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    fname = 'sipp_raw_data_UFF_Perdue_02_10_reduced_1.csv'
    fpath = dir + fname
    inputPacketsCols = ['CallRate(P)']
    droppedPacketsCols = ['FailedCall(P)']
    interval = 1
    calls_to_packets = calls_to_packets
    
    df = parseDataFile(fpath, inputPacketsCols, droppedPacketsCols)
    
    increment = 50
    if interval == -1:
        interval = df.shape[0]-1
    
    totalInputPackets = 0
    totalDroppedPackets = 0
    MSE = 0
    intervalCount = 0
    print 'time_interval mu lambda empirical_PK estimated_PK squareLoss'
    est_PKs = []
    emp_PKs = []
    my_lambdas = []
    timeIntervals = []
    squaredLoss = 0
    skipFlag = False
    for i, row in df.iterrows():
        if i % interval == 0 and i != 0:
            intervalCount += 1
            if intervalCount == 194:
                dbg = 1
            
            avgDroppedPackets = totalDroppedPackets / float(interval)
            avgInputPackets   = totalInputPackets / float(interval)
            
            my_lambda = avgInputPackets
            my_lambdas.append(my_lambda)
            
            my_mu = mu
            rho = my_lambda / my_mu
            
            try:
                empirical_PK = float(totalDroppedPackets) / totalInputPackets
            except:
                #empirical_PK = 0.0
                skipFlag = True
            if (totalDroppedPackets == 0 and totalInputPackets == 0) or totalInputPackets == 0:
                #empirical_PK = 0
                skipFlag = True
            
            try:
                estimated_PK = math.exp(solver.M_M_m_K_log(rho, m, K))
                #estimated_PK = solver.f(rho, m, K)
            except:
                #estimated_PK = 0.0
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
            
            squaredLoss = (empirical_PK - estimated_PK)**2
            MSE += squaredLoss
            
            print intervalCount, mu, my_lambda, empirical_PK, estimated_PK, squaredLoss
            
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
    
    print 'MSE=', squaredLoss/intervalCount
    
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
    
    
    
    '''
    #drawing mu vs lambda
    plt.ylabel('Packet/s')
    plt.xlabel('Time Interval')
    lines = plt.plot(timeIntervals, [mu for i in timeIntervals], '--r' ,label='Mu') 
    plt.setp(lines, linewidth=2.0)
    lines = plt.plot(timeIntervals, my_lambdas, 'b' ,label='Lambda') 
    plt.setp(lines, linewidth=2.0)
    '''
    
    
    #drawing lambda Vs. estimated, empirical PK
    '''
    plt.ylabel('Probability')
    plt.xlabel('Packets/s)
    lambda_PK = sorted(zip(my_lambdas, est_PKs))
    sortedLambdas = [y[0] for y in lambda_PK]
    PKs = [y[1] for y in lambda_PK]
    lines = plt.plot(sortedLambdas, PKs, '--r' ,label='Estimated PK')
    plt.setp(lines, linewidth=2.0)
    
    lambda_PK = sorted(zip(my_lambdas, emp_PKs))
    sortedLambdas = [y[0] for y in lambda_PK]
    PKs = [y[1] for y in lambda_PK]
    lines = plt.plot(sortedLambdas, PKs, 'b' ,label='Empirical PK')  
    plt.setp(lines, linewidth=2.0)
    '''
    
    plt.legend(loc = 2, prop={'size':17}, labelspacing=0.1) 
    fig.suptitle(fname, fontsize=12, fontweight='bold', horizontalalignment='center', y=.86)
    plt.grid()                                                                     
    #plt.savefig(resultsPath+'combined_rec_prec_plot_withActionSampling.pdf', bbox_inches='tight')
    plt.show() 
   

def calculate_mu_lambda(fname):
    inputPacketsCols = ['CallRate(P)']
    droppedPacketsCols = ['FailedCall(P)']
    
    calls_to_packets = 6.0
    interval = 1
    
    m = 1
    K = 972
    
    df = parseDataFile(fname, inputPacketsCols, droppedPacketsCols)
    
    totalInputPackets = 0
    totalDroppedPackets = 0
    my_eps = 0
    
    my_lambdas = []
    my_mus = []
    timeIntervals = []
    timeInterval = 0
    
    if interval == -1:
        interval = df.shape[0]-1
    for i, row in df.iterrows():
        if i % interval == 0 and i != 0:
        
            avgInputPackets   = totalInputPackets / float(interval)
            
            my_lambda = avgInputPackets
            PK = float(totalDroppedPackets + my_eps) / totalInputPackets
            if math.isnan(PK):
            
                totalInputPackets = 0
                totalDroppedPackets = 0
                    
                for inpCol in inputPacketsCols:
                    totalInputPackets += row[inpCol]*calls_to_packets
                
                for dropCol in droppedPacketsCols:
                    totalDroppedPackets += row[dropCol]*calls_to_packets
                
                continue
                
            
            try:
                rho_0 = 0.1
                rho = float( fsolve(solver.M_M_m_K_log_solve, rho_0, (m, K, PK))[0] )
                #rho = fsolve(solver.M_M_m_K_solve, rho_0, (m, K, PK))
                #rho = float( fsolve(solver.M_M_m_K_solve, rho_0, (m, K, PK))[0] )
            except:
                totalInputPackets = 0
                totalDroppedPackets = 0
                
                for inpCol in inputPacketsCols:
                    totalInputPackets += row[inpCol]*calls_to_packets
                
                for dropCol in droppedPacketsCols:
                    totalDroppedPackets += row[dropCol]*calls_to_packets
                
                continue
            
            timeInterval += 1
            timeIntervals.append(timeInterval)
            
            mu = my_lambda / rho
            
            pkk = math.exp(solver.M_M_m_K_log(rho, m, K))
            
            my_lambdas.append(my_lambda)
            my_mus.append(mu)
            
            totalInputPackets = 0
            totalDroppedPackets = 0
           
            for inpCol in inputPacketsCols:
                totalInputPackets += row[inpCol]*calls_to_packets
                
            for dropCol in droppedPacketsCols:
                totalDroppedPackets += row[dropCol]*calls_to_packets
           
            
        else:
            
            for inpCol in inputPacketsCols:
                totalInputPackets += row[inpCol]*calls_to_packets
                
            for dropCol in droppedPacketsCols:
                totalDroppedPackets += row[dropCol]*calls_to_packets
            
    
    return my_mus, my_lambdas
   
def estimate_mu_within_minRate_and_maxRate(dir, summaryFile, minFailRate, maxFailRate, fileName2mu):
    sfile = dir+summaryFile
    df = pd.read_csv(sfile, usecols = ['Rate File', ' Failed Calls'])
    df.fillna(0, inplace=True)
    inputRates = []
    serviceRates = [] 
    for i, row in df.iterrows():
        if row[' Failed Calls'] < minFailRate or row[' Failed Calls'] > maxFailRate:
            continue
        fname = 'sipp_data_' + row['Rate File'] + '_1.csv'
        simulationFile = dir + fname #sipp_data_UFF_Perdue_01_1_reduced_1.csv     UFF_Perdue_01_12_reduced
        if fname in fileName2mu:
            serviceRates.append(fileName2mu[fname])
            continue
        mus, lambdas = calculate_mu_lambda(simulationFile)
        if len(lambdas) > 0 and len(mus) > 0:
            inputRates.append(max(lambdas))
            serviceRates.append(max(mus))
            fileName2mu[fname] = max(mus)
            
        #print fname, i, len(df), max(lambdas), max(mus)
    #print 'Avg. lambda = ', sum(inputRates)/float(len(inputRates))
    #print 'Avg. mu     = ', sum(serviceRates)/float(len(serviceRates))
    return sum(serviceRates)/float(len(serviceRates))
        
def plot_mu_vs_minFailRate(dir, summaryFile):
    fileName2mu = {}
    for z in [0, 50, 100, 250,500,750,1000,1250,1500,1750,2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750, 10000]:
        mu = estimate_mu_within_minRate_and_maxRate(dir, summaryFile, z, 1e10,fileName2mu)
        print z,mu
        
def plot_mu_vs_maxFailRate(dir, summaryFile):
    fileName2mu = {}
    for maxDropRate in [5, 25, 50, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]:
        mu = estimate_mu_within_minRate_and_maxRate(dir, summaryFile, 1, maxDropRate,fileName2mu)
        print maxDropRate,mu

def plot_lambda_mu():
    #using server data
    '''
    #dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/Const_Rate/'
    #fpath = dir + 'results_1_2018.09.25-19.46.08/kamailio_results/kamailio_data_const_rate_1200_1800_10_50_10.csv'
    
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_1_2018.09.28-19.40.18/'
    fpath = dir + '/kamailio_results/kamailio_data_const_rate_1200_1800_10_50_10.csv'
    
    inputPacketsCols = ['core:rcv_replies', 'core:rcv_requests']
    droppedPacketsCols = ['sl:404_replies', 'System:Packet_Drops']
    #404_replies = server get the packet but has capacity to processed (has no relation with size of queue, CPU is overloaded)
    #System:Packet_Drops = no buffer in queue
    '''
    
    #using client data
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    fpath = dir + 'sipp_raw_data_UFF_Perdue_07_44_reduced_1.csv'
    inputPacketsCols = ['CallRate(P)']
    droppedPacketsCols = ['FailedCall(P)']
    
    calls_to_packets = 6.0
    interval = 1
    increment = 10
    #startRate = 1200
    #endRate = 1800
    
    m = 1
    K = 972
    
    
    df = parseDataFile(fpath, inputPacketsCols, droppedPacketsCols)
    
    totalInputPackets = 0
    totalDroppedPackets = 0
    my_eps = 0
    
    my_lambdas = []
    my_mus = []
    calls_per_sec = []
    timeIntervals = []
    timeInterval = 0
    
    if interval == -1:
        interval = df.shape[0]-1
    print 'empirical PK, estimated PK, my_lambda, mu'
    for i, row in df.iterrows():
        if i % interval == 0 and i != 0:
            
            #avgDroppedPackets = totalDroppedPackets / float(interval)
            avgInputPackets   = totalInputPackets / float(interval)
            
            my_lambda = avgInputPackets
            PK = float(totalDroppedPackets + my_eps) / totalInputPackets
            if math.isnan(PK):
                print PK
                totalInputPackets = 0
                totalDroppedPackets = 0
                
                
                for inpCol in inputPacketsCols:
                    totalInputPackets += row[inpCol]*calls_to_packets
                
                for dropCol in droppedPacketsCols:
                    totalDroppedPackets += row[dropCol]*calls_to_packets
                
                #startRate += increment
                continue
                
    
            
            try:
                rho_0 = 0.1
                rho = float( fsolve(solver.M_M_m_K_log_solve, rho_0, (m, K, PK))[0] )
                #rho = fsolve(solver.M_M_m_K_solve, rho_0, (m, K, PK))
                #rho = float( fsolve(solver.M_M_m_K_solve, rho_0, (m, K, PK))[0] )
            except:
                print PK
                totalInputPackets = 0
                totalDroppedPackets = 0
                
                
                for inpCol in inputPacketsCols:
                    totalInputPackets += row[inpCol]*calls_to_packets
                
                for dropCol in droppedPacketsCols:
                    totalDroppedPackets += row[dropCol]*calls_to_packets
                
                #startRate += increment
                continue
            
            timeInterval += 1
            timeIntervals.append(timeInterval)
            
            mu = my_lambda / rho
            
            #pkk = solver.M_M_m_K(rho, m, K)
            pkk = math.exp(solver.M_M_m_K_log(rho, m, K))
            
            my_lambdas.append(my_lambda)
            my_mus.append(mu)
            
            print PK, ',', pkk, ',', int(my_lambda), ',', int(mu) 
            #print 'using M/M/m/K \nrho_0=%.5f, rho_final=%.5f, PK\'=%.5f, PK\'-PK=%.5f, mu=%.5f, lambda=%.5f' % (rho_0, rho, pkk, diff, mu, my_lambda)
            #startRate += increment
            
            totalInputPackets = 0
            totalDroppedPackets = 0
           
            for inpCol in inputPacketsCols:
                totalInputPackets += row[inpCol]*calls_to_packets
                
            for dropCol in droppedPacketsCols:
                totalDroppedPackets += row[dropCol]*calls_to_packets
           
            
        else:
            
            for inpCol in inputPacketsCols:
                totalInputPackets += row[inpCol]*calls_to_packets
                
            for dropCol in droppedPacketsCols:
                totalDroppedPackets += row[dropCol]*calls_to_packets
            
            
    print 'Max mu = ', max(my_mus)
    print 'Min mu = ', min(my_mus)
    print 'Avg mu = ', float(sum(my_mus))/len(my_mus)

    fig = plt.figure(1, figsize=(6, 4))
    #plt.xticks([x for x in range(1,max(timeIntervals))])
    axes = plt.gca()
    #axes.set_ylim([-0.05,0.6])
    plt.ylabel('Packet/s')
    plt.xlabel('Time Interval')
    lines = plt.plot(timeIntervals, my_mus, '--r' ,label='Mu') 
    plt.setp(lines, linewidth=2.0)
    lines = plt.plot(timeIntervals, my_lambdas, 'b' ,label='Lambda') 
    plt.setp(lines, linewidth=2.0)
    plt.legend(loc = 2, prop={'size':17}, labelspacing=0.1) 
    fig.suptitle('Lambda Vs. estimated Mu', fontsize=15, fontweight='bold', horizontalalignment='center', y=.86)
    plt.grid()                                                                     
    #plt.savefig(resultsPath+'combined_rec_prec_plot_withActionSampling.pdf', bbox_inches='tight')
    plt.show() 




if __name__ == "__main__":
    mpl.rcParams.update({'font.size': 17})
    dir = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/projects/queueing/results_24_2018.10.20-13.31.38_client_server/sipp_results/'
    summaryFile = 'summary_data_dump.csv'
    #estimate_mu(dir, summaryFile, minFailRate = 1000)
    #plot_lambda_mu()
    #estimate_mu_within_minRate_and_maxRate(dir, summaryFile, 1, 100, {})
    #plot_mu_vs_maxFailRate(dir, summaryFile)
    PK_estimation_lambdaFromData(K=1, m=1, mu=850.48, scalingThreshold = 0.005, calls_to_packets = 1.0)
    print('DONE!')