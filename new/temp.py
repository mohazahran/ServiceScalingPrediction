import numpy as np
import torch
import function as F
import data
import module
np.set_printoptions(precision=8, suppress=True)

seed = 47
data_kargs = dict(k=20, m=1, const_mu=25, epsilon=1e-4, ind=[0, 1], focus=-1)
train_data = data.DataMMmK(n=200, lamin=10, lamax=40, seed=seed - 1, **data_kargs)
test_data  = data.DataMMmK(n=400, lamin=1 , lamax=50, seed=seed + 1, **data_kargs)

model = module.MMmKModule(train_data, method='rrx', trick='hav')
# model = module.MMmKModule(train_data, method='pow')
task = module.Task(train_data, test_data, model, ctype='resi', alpha=1e6, seed=seed)
task.fit_from_rand(100)