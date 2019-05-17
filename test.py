import torch
import function as F
import numpy as np
np.set_printoptions(precision=8, suppress=True)

X = torch.from_numpy(np.random.randint(0, 10, size=(11, 11))).float()
X.requires_grad = False
pos = [(i + 1, i) for i in range(X.size(0) - 1)]
for i in range(X.size(0)):
    X.data[i, i] = 0

np.random.seed(47)
X1 = X.clone()
X1.requires_grad = True
pi = F.stdy_dist_rrx(X1, None)
loss = pi.sum()
loss.backward()
# // print(X1.grad.data.numpy() * 1e6)

np.random.seed(47)
X2 = X.clone()
X2.requires_grad = True
pi = F.optim_stdy_dist_rrx(X2, None, pos)
loss = pi.sum()
loss.backward()
# // print(X2.grad.data.numpy() * 1e6)

dev = 0
for i, j in pos:
    t = (X2.grad.data[i, j] - X1.grad.data[i, j]).item()
    dev += t
assert dev == 0