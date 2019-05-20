import sys
import os
import data
import module

# number of training samples
rng_lst = [
    'l',
    's',
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
case = sys.argv[1]
num_epochs = 300
alpha = 100
lr = 1

# create saving folder
root = "{}_cond".format(case)
if os.path.isdir(root):
    pass
else:
    os.makedirs(root)

# traverse number of training samples
for rng in rng_lst:
    # load data
    seed, train_data, test_data = getattr(data, case)(rng)
    print(len(train_data))
    print(len(test_data))
    
    # allocate model
    for magic in magic_lst:
        model = module.MMmKModule(train_data, **magic)
        name = "{}_{}_cond_{}".format(case, rng, magic['trick'])
        task = module.Task(train_data, test_data, model, ctype='cond', alpha=alpha, seed=seed, lr=lr)
        task.fit_from_rand(num_epochs, root=root, name=name)