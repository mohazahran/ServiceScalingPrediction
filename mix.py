import sys
import numpy as np
import matplotlib.pyplot as plt

n = int(sys.argv[1])
v = int(sys.argv[2])
X = np.zeros(shape=(n, n))
for i in range(n - 1):
    X[i + 1, i] = v - 1
    X[i, i + 1] = v
Q = X - np.diag(np.sum(X, axis=1, keepdims=False))
P = np.eye(n) + Q / (np.max(np.fabs(Q)) + 0.001)
u, _ = np.linalg.eig(P)
e2 = sorted(u)[-2]
elst, dlst = [], []
for i in range(1, 1024 + 1):
    E = np.linalg.matrix_power(P, i)
    pi = np.mean(E, axis=0, keepdims=True)
    mix = max([np.sum(np.square(E[i] - pi)) for i in range(n)])
    if mix < 1e-6:
        print(i)
        exit()