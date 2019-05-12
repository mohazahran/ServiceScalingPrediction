import data
import module


# numerical method list
magic_lst = [
    {'method': 'pow', 'trick': 'pow'  },
    {'method': 'rrx', 'trick': 'rrinf'},
    {'method': 'rrx', 'trick': 'inf'  },
    {'method': 'rrx', 'trick': 'rr'   },
]

# alpha strength list
alpha_lst = [
    '1'
]

# generate data
case = 'mm1k'
num = 100
seed, train_data, test_data = getattr(data, case)(num)

# criterion
ctype = 'cond'

# traverse grid
num_epochs = 300
for magic in magic_lst:
    for alpha_str in alpha_lst:
        alpha = float(alpha_str)
        if alpha < 0:
            model = module.QueueModule(train_data, **magic)
        else:
            model = module.MMmKModule(train_data, **magic)
        name = "{}_{}_{}_{}_{}".format(case, num, ctype, magic['trick'], alpha_str)
        task = module.Task(train_data, test_data, model, ctype=ctype, alpha=alpha, seed=seed)
        task.fit_from_rand(num_epochs, name=name)