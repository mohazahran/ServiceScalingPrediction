'''
Created on Dec 13, 2018

@author: mohame11
'''

import solver
import math


def main():
    K = 5
    m = 3
    theLambda = 45.0
    theMu = 50.0
    
    rho = float(theLambda) / float(theMu)
    
    logPK = solver.M_M_m_K_log(rho, m, K)
    print math.exp(logPK)
    
    




if __name__ == "__main__":
    main()
    print('DONE!')
