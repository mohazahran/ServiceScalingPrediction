import sys
import os
import data
import module

# number of training samples
num_lst = [
    1,
    5,
    10,
    25,
    50,
    100,
    200,
]

# numerical method list
magic_lst = [
    {'method': 'pow', 'trick': 'pow'  },
    {'method': 'rrx', 'trick': 'rrinf'},
    {'method': 'rrx', 'trick': 'inf'  },
    {'method': 'rrx', 'trick': 'rr'   },
]

# constant settings
case = 'mmmmr'
ctype = 'cond'
num_epochs = 300
alpha_str = sys.argv[1]

# create saving folder
root = "{}_{}_{}_{}_{}".format(case, 'all', ctype, 'all', alpha_str)
if os.path.isdir(root):
    pass
else:
    os.makedirs(root)

# traverse number of training samples
for num in num_lst:
    # generate data
    seed, train_data, test_data = getattr(data, case)(num)

    # traverse numerical method
    for magic in magic_lst:
        alpha = float(alpha_str)
        if alpha < 0:
            model = module.QueueModule(train_data, **magic)
        else:
            model = module.MMmKModule(train_data, **magic)
        name = "{}_{}_{}_{}_{}".format(case, num, ctype, magic['trick'], alpha_str)
        task = module.Task(train_data, test_data, model, ctype=ctype, alpha=alpha, seed=seed)
        task.fit_from_rand(num_epochs, root=root, name=name)