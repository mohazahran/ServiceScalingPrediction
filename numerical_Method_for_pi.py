import numpy as np

# solve for x for : linear matrix equation ax = b.

a = np.array([[0.6, 0.2, 0.3],[0.3, 0.5, 0.2],[0.8, 0.1, 0.1]])
p = a - np.eye(3,3)
p_ = np.concatenate((p,np.ones((3,1))), axis = 1)
b = np.zeros((1,4))
b[-1][-1] = 1
ppi = np.array([[0.558, 0.26, 0.182]])
pi, residues, rank, s =   np.linalg.lstsq(p_.T, b.T)
print pi
#np.linalg.solve(p_.T, b.T)



#from numpy import diag, linspace, ones, eye
#from krypy.linsys import LinearSystem, Minres

# construct the linear system Ax = b
#A = diag(linspace(1, 2, 20))
#A[0, 0] = -1e-5
#c = ones(20)
#linear_system = LinearSystem(A, c, self_adjoint=True)
#linear_system = LinearSystem(p_.T, b.T, self_adjoint=True)

# solve the linear system (approximate solution is solver.xk)
#solver = Minres(linear_system)