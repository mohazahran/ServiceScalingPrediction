import sys
import os
import threading
import itertools


r"""
Thread
======
- **LibThread** : Lib Thread Holder
"""


class LibThread(threading.Thread):
    r"""Lib Thread Holder"""
    def __init__(self, cmd):
        r"""Initialize the class

        Args
        ----
        cmd : str
            Lib command line.

        """
        # super call
        threading.Thread.__init__(self)

        # save necessary attributes
        self.cmd = cmd

    def run(self):
        r"""Operate thread"""
        # output arguments
        print("< {}".format(self.cmd))
        os.system(self.cmd)
        print("> {}".format(self.cmd))


if __name__ == '__main__':
    r"""Main Entrance"""
    # parse arguments
    task, = sys.argv[1:]

    # ensure logging folder
    if os.path.isdir('log'):
        pass
    else:
        os.makedirs('log')

    # ensure task result folder
    if os.path.isdir(task):
        pass
    else:
        os.makedirs(task)

    # traverse all threads
    thread_lst = []
    num_lst       = ['400', '200', '100', '50', '25']
    dtype_lst     = ['sym', 'raw', 'pow']
    ctype_lst     = ['resi', 'cond', 'mse']
    alpha_str_lst = ['0', '1', '1000']
    hyper_combs = itertools.product(num_lst, dtype_lst, ctype_lst, alpha_str_lst)
    for combine in hyper_combs:
        num, dtype, ctype, alpha_str = combine
        args = ('python', 'lib.py', task, num, dtype, ctype, alpha_str)
        name = '_'.join([str(itr) for itr in args[2:]])
        cmd = ''.join(["{:<7s}".format(str(itr)) for itr in args])
        cmd = "{} > {} 2>&1".format(cmd, os.path.join('log', "{}.log".format(name)))
        thread_lst.append(LibThread(cmd))

    # start and wait all threads
    for itr in thread_lst:
        itr.start()
    for itr in thread_lst:
        itr.join()