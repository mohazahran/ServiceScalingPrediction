import sys
import os
import data
import module

# number of training samples
rng_lst = [
    's',
    'l',
]
num_lst = [
    200,
    100,
    50,
    25,
    10,
    5,
]

# numerical method list
magic_lst = [
    {'method': 'rrx', 'trick': 'rrinf'},
    {'method': 'pow', 'trick': '7'    },
    {'method': 'pow', 'trick': '4'    },
    {'method': 'rrx', 'trick': 'inf'  },
    {'method': 'rrx', 'trick': 'rr'   },
]

# constant settings
case = 'mm1k'
ctype = 'cond'
num_epochs = 1
alpha_str = sys.argv[1]

# create saving folder
root = "{}_{}_{}_{}_{}".format(case, 'all', ctype, 'all', alpha_str)
if os.path.isdir(root):
    pass
else:
    os.makedirs(root)

# traverse number of training samples
for rng in rng_lst:
    for num in num_lst:
        # generate data
        seed, train_data, test_data = getattr(data, case)(rng, num)
    
        # traverse numerical method
        for magic in magic_lst:
            alpha = float(alpha_str)
            if alpha < 0:
                model = module.QueueModule(train_data, **magic)
            else:
                model = module.MMmKModule(train_data, **magic)
            name = "{}_{}_{}_{}_{}_{}".format(case, rng, num, ctype, magic['trick'], alpha_str)
            task = module.Task(train_data, test_data, model, ctype=ctype, alpha=alpha, seed=seed)
            task.fit_from_rand(num_epochs, root=root, name=name)