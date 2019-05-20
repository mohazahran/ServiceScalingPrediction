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
num_epochs = 500
alpha = 100
lr = 1

# create saving folder
root = 'real_cond'
if os.path.isdir(root):
    pass
else:
    os.makedirs(root)

# traverse number of training samples
for rng in rng_lst:
    # load data
    seed, train_data, test_data = data.real(rng)
    print(len(train_data))
    print(len(test_data))
    
    # allocate model
    for magic in magic_lst:
        model = module.MMmKModule(train_data, **magic)
        name = "real_{}_cond_{}".format(rng, magic['trick'])
        task = module.Task(train_data, test_data, model, ctype='cond', alpha=alpha, seed=seed, lr=lr)
        task.fit_from_rand(num_epochs, root=root, name=name)