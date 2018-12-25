'''
Created on Dec 2, 2018

@author: mohame11
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import solver


class MC_steady(nn.Module):
    def __init__(self, numOfStates):
        super(MC_steady, self).__init__()
        self.numOfStates = numOfStates
        #self.steadyStateDist = nn.Parameter(torch.rand(self.numOfStates, 1),requires_grad = True)
        self.steadyStateDist = torch.FloatTensor([[0.3,0.3,0.4]])
        #self.objective = nn.MSELoss(size_average = True)
        #self.optimiser = torch.optim.Adam(self.parameters(), lr = 0.01)
        
    def trainSteadyState(self, MC_transitions):
        MC_transitions = torch.FloatTensor(MC_transitions)
        for i in range(MC_transitions.size(0)):
            MC_transitions[i] /= torch.sum(MC_transitions[i])
        
        print MC_transitions
        #startState = torch.FloatTensor([[1,0,0,0,0]])
        #MC_transitions = F.softmax(MC_transitions)
        #MC_transitions2 = torch.FloatTensor(MC_transitions)
        while True:
            #self.steadyStateDist = torch.mm(self.steadyStateDist, MC_transitions)
            MC_transitions = torch.mm(MC_transitions, MC_transitions)
            #stateDist = torch.mm(startState, MC_transitions)
            #stateDist = torch.mm(MC_transitions, self.steadyStateDist)
            #loss = self.objective(stateDist, self.steadyStateDist)
            #loss.backward()
            #self.optimiser.step()
            #self.optimiser.zero_grad()
            #print self.steadyStateDist
            print MC_transitions
            print
            #print stateDist
            
    
    
    



def main():

    lmbda = 40.0
    mu = 50.0
    m = 1
    #for MM1
    rho = lmbda/mu
    p0 = 1 - rho
    K = 5
    '''
    print p0,',',
    for i in range(1,K):
        pi = p0 * (rho**i)
        print pi,',',
    '''
    pi = []
    for n in range(0,K+1):
        pn = solver.M_M_m_K_getProbAt_k_2(rho, m, K, n)
        pi.append(pn)
        
    #MC_transitions = [ [ 0.18, 0.42, 0.40] , [0.05, 0.40, 0.55] , [0.10, 0.37, 0.53] ]
    #MC_transitions = [[0,lmbda,0,0,0], [mu,0,lmbda,0,0], [0,2*mu,0,lmbda,0] ,[0,0,3*mu,0,lmbda] ,[0,0,0,3*mu,0] ]
    
    print pi
    
    #MC_transitions = [[1-lmbda,lmbda,0,0,0], [mu,1-(mu+lmbda),lmbda,0,0], [0,mu,1-(mu+lmbda),lmbda,0] ,[0,0,mu,1-(mu+lmbda),lmbda] ,[0,0,0,mu,1-mu] ]
    trans_rate = [[-lmbda,lmbda,0,0,0], [mu,-(mu+lmbda),lmbda,0,0], [0,mu,-(mu+lmbda),lmbda,0] ,[0,0,mu,-(mu+lmbda),lmbda] ,[0,0,0,mu,-mu] ]
    
    Q = torch.FloatTensor(trans_rate)
    pi = torch.FloatTensor(pi)
    p = torch.mm(pi, Q)
    
    '''
    MC_probs = [[0,1,0,0,0], 
                [mu/(mu+lmbda),0,lmbda/(mu+lmbda),0,0], 
                [0,mu/(mu+lmbda),0,lmbda/(mu+lmbda),0],
                [0,0,mu/(mu+lmbda),0,lmbda/(mu+lmbda)],
                [0,0,0,1,0] ]
    '''
    
    #from scipy.linalg import expm
    #MC_transitions = expm(trans_rate)
    
    #MC_transitions = F.softmax(torch.FloatTensor(trans_rate))
    
    #MC_transitions = [[0, 0.9,0.1], [1, 0, 0], [0.6,0.4,0]]
    #MC_transitions = [[0.6, 0.4, 0], [0.6, 0, 0.4], [0 , 1, 0]]
    #MC_transitions = [[0.6,0.4,0,0], [0.6,0,0.4,0], [0,0.6,0,0.4], [0,0,0.6,0.4]]
    mc = MC_steady(2)
    mc.trainSteadyState(MC_probs)





if __name__ == "__main__":
    main()
    print('DONE!')
    
        