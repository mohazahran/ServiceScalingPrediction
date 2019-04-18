'''
Created on Sep 3, 2018

@author: mohame11
'''
import math
import numpy as np
from scipy.optimize import *
import random


def bith_death_closedForm(lambdas, mus, k):
    '''
    lambdas: ids 0->K-1
    mus: ids 1->K
    '''
    K = len(mus)
    
    sumRhos = 0.0
    for j in range(1, k):
        rhos = 1.0
        for i in range(j):
            rhos *= lambdas[i]/mus[i]
        sumRhos += rhos
            
    
    
    p0 = 1.0/(1.0+sumRhos)
    
    lambdasProd = 1.0
    musProd = 1.0
    
    for i in range(k):
        lambdasProd *= lambdas[i]
        musProd *= mus[i]
        
    
    pk = (lambdasProd/musProd)*p0
    return pk


def M_M_m_K_simulation(la, mu, m, K, steps):
    x = 0  # current state (#of alive requests)
    states = [str(x)]
    #states = []
    for i in range(steps):
        my_lambda = la
        my_mu = mu

        if x == 0:
            my_mu = 0
        if x == K:
            my_lambda = 0

        if x <= m:
            my_mu = my_mu * x
        else:
            my_mu = my_mu * m

        rate = my_lambda + my_mu
        # rate now is a normalizing const
        # so that with prob lambda/rate we move to next state
        # and with prob mu/rate we go back to previous state
        rnd = random.random()
        if rnd >= float(my_lambda) / float(rate):
            x -= 1
        else:
            x += 1

        #if x != 0:
        states.append(str(x))
        # t += math.log(random.random())/rate

    statesCount = len(set(states))
    return states


def M_M_m_K_getProbAt_k_2(rho, m, K, n):
    p0 = 0.0
    p0_part1 = 0.0
    for i in range(0,m+1):
        p0_part1 += (rho**i)/math.factorial(i)
    
    p0_part2 = 0.0
    for i in range(1,K-m+1):
        p0_part2 += (rho/m)**i
    p0_part2 = p0_part2 * (rho**m)/math.factorial(m)
    
    p0 = 1.0/(p0_part1 + p0_part2)
    
    if n == 0:
        return p0
    elif n <= m:
        pn = p0 * (rho**n)/math.factorial(n)
    else:
        pn = p0 * (rho**n)/math.factorial(n) * (rho/m)**(n-m)
    
    return pn
        
        
 
def M_M_m_K_log(x, m, K):
    #print x
    c1 = K * math.log(x)
    c2 = math.log(math.factorial(m))
    c3 = (K-m) * math.log(m)
    logC_K = c1 - c2 - c3
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
    part3 = part3 / float(math.factorial(m))
        
    logP0 = -1 * math.log(1.0 + part2 + part3)
    
    logP_K = logC_K + logP0
    
    return logP_K   

def M_M_m_K_getProbAt_k(rho, m, K, n):
    cn = 0.0
    if n < m:
        cn = (rho ** n)/math.factorial(n)
    elif m<=n and n<= K:
        cn = (rho ** n) / (math.factorial(m) * (m**(n-m)) )
    else:
        cn = 0.0
        
    p0 = 0.0
    if n < m:
        summ = 0.0
        for i in range(1,m):
            summ += (rho ** i) / math.factorial(i)
        p0 = 1.0/(1.0+summ)
        
    elif m<=n and n<= K:
        part2 = 0.0
        for i in range(1,m):
            part2 += (rho ** i) / math.factorial(i)
            
        part3 = 1.0/math.factorial(m)
        for i in range(m,K+1):
            part3 += (rho ** i) / (m**(i-m))
        
        p0 = 1.0 / (1.0 + part2 + part3)
    
    pn = p0*cn
    return pn
        
        
        
        
            
    
    

def M_M_1_K_log_solve(x, K, PK):
    print x
    logp = math.log(1-x) + K*math.log(x) - math.log(1-(x**(K+1)))
    return logp - math.log(PK)

def M_M_1_K_solve(x, K, PK):
    p = (1.0-x) * (x**K) / (1.0 - (x**(K+1)))
    return p - PK


def M_M_1_K(x, K):
    p = (1.0-x) * (x**K) / (1.0 - (x**(K+1)))
    return p




def M_M_m_K_log_solve(x, m, K, PK):
    #print x
    c1 = K * math.log(x)
    c2 = math.log(math.factorial(m))
    c3 = (K-m) * math.log(m)
    logC_K = c1 - c2 - c3
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
    part3 = part3 / float(math.factorial(m))
        
    logP0 = -1 * math.log(1.0 + part2 + part3)
    
    logP_K = logC_K + logP0
    
    return logP_K - math.log(PK)


def f(x, m, K):
    #print x
    part1 = (x**K) / (math.factorial(m) * m**(K-m))
    
    #part1 = K * math.log(x) - math.log((math.factorial(m) * m**(K-m)))
    #part1 = math.exp(part1) 
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
        
    part3 *= 1.0/math.factorial(m)
      
    tot = part1 * (1.0/(1.0+part2+part3))
    #tot = part1 + math.log(1.0/(1.0+part2+part3))
    #tot = math.exp(tot) 
    
    return tot
    #return tot - PK


def M_M_m_K(x, m, K):
    #print x
    part1 = (x**K) / (math.factorial(m) * m**(K-m))
    
    #part1 = K * math.log(x) - math.log((math.factorial(m) * m**(K-m)))
    #part1 = math.exp(part1) 
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
        
    part3 *= 1.0/math.factorial(m)
      
    tot = part1 * (1.0/(1.0+part2+part3))
    #tot = part1 + math.log(1.0/(1.0+part2+part3))
    #tot = math.exp(tot) 
    
    #return tot
    return tot

def M_M_m_K_solve(x, m, K, PK):
    #print x
    part1 = (x**K) / (math.factorial(m) * m**(K-m))
    
    #part1 = K * math.log(x) - math.log((math.factorial(m) * m**(K-m)))
    #part1 = math.exp(part1) 
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
        
    part3 *= 1.0/math.factorial(m)
      
    tot = part1 * (1.0/(1.0+part2+part3))
    #tot = part1 + math.log(1.0/(1.0+part2+part3))
    #tot = math.exp(tot) 
    
    #return tot
    return tot - PK




def main():
    m = 1 # #workers
    K = 972 # queue size in terms of #request
    my_lambda = 8400 #arrival per unit time
    PK = 1775.0/124259 #prob of failure
    #PK = 0.001
    rho_0 = 0.1 #lambda/mu (initial point to start)
    #mu Amit's intuition is 8400 packet/sec
    
    print 'PK=', PK
    
    rho = fsolve(M_M_m_K_log_solve, rho_0, (m, K, PK)) #solve for rho
    
    diff = M_M_m_K_log_solve(rho, m, K, PK)
    pkk = f(rho, m, K, PK)
    mu = my_lambda / rho
    print 'using M/M/m/K \nrho_0=%.5f, rho_final=%.5f, PK\'=%.5f, PK\'-PK=%.5f, mu=%.5f, lambda=%.5f' % (rho_0, rho, pkk, diff, mu, my_lambda)
    
    '''
    rho = fsolve(M_M_1_K_log_solve, rho_0, (K, PK)) #solve for rho
    diff = M_M_1_K_log_solve(rho, K, PK)
    pkk = M_M_1_K(rho, K)
    mu = my_lambda / rho
    print 'using M/M/1/K\nrho_0=%.5f, rho_final=%.5f, PK\'=%.5f, PK\'-PK=%.5f, mu=%.5f' % (rho_0, rho, pkk, diff, mu)
    '''
    
     
    #sol = newton(f, x, (m, K, PK))
    #diff = f(sol, m, K, PK)
    #print 'using newton\nrho_0=%.5f, rho_final=%.5f, PK\'-PK=%.5f' % (x, sol, diff)
    

    
def testing():
    true_m = 3
    true_K = 5
    inp = 15.0
    true_mu = 50.0 
    rho = 9.88
    pks1 = []
    pks2 = []
    for n in range(0,6):
        pk1 = M_M_m_K_getProbAt_k(rho, true_m, true_K, n)
        pks1.append(pk1)
        pk2 = M_M_m_K_getProbAt_k_2(rho, true_m, true_K, n)
        pks2.append(pk2)
        print pk1, pk2
    print '\n', sum(pks1), sum(pks2)
    pk = M_M_m_K_log(rho, true_m, true_K)
    print math.exp(pk)



def testing_MMmK_simulation():
    la = 10
    mu = 12
    K = 5
    m = 3
    steps = 100
    states = M_M_m_K_simulation(la, mu, m, K, steps)
    PK_ = states.count(str(K)) / float(len(states))

    #print states
    print 'est_PK=', PK_
    print math.exp(M_M_m_K_log(float(la) / float(mu), m, K))
    print M_M_m_K(float(la) / float(mu), m, K)



if __name__ == "__main__":
    testing_MMmK_simulation()
    #testing()
    #main()
    print('DONE!')