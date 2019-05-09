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
    holder, task, hyper = sys.argv[1:4]
    assert holder in ('lib.py', 'press.py')
    assert task in ('mmmmr', 'lbwb', 'cio')
    assert hyper in ('quick', 'hyper')
    if holder == 'press.py':
        M = int(sys.argv[4])
    else:
        pass

    # ensure logging folder
    if os.path.isdir('log'):
        pass
    else:
        os.makedirs('log')

    # ensure task result folder
    if holder == 'press.py':
        root = "press-{}-{}".format(M, task)
    else:
        root = "{}-{}".format(hyper, task)
    if os.path.isdir(root):
        pass
    else:
        os.makedirs(root)

    # traverse all threads
    thread_lst = []
    num_lst = ['400', '200', '100', '50', '25', '1']
    for num in num_lst:
        if holder == 'press.py':
            args = ('python', holder, task, num, hyper, M)
            name = '_'.join([str(itr) for itr in args[2:-2]])
            name = "{}_press_{}".format(name, M)
        else:
            args = ('python', holder, task, num, hyper)
            name = '_'.join([str(itr) for itr in args[2:]])
        cmd = ' '.join(["{:<8s}".format(str(itr)) for itr in args])
        cmd = "{} 2>&1 > {}".format(cmd, os.path.join('log', "{}.log".format(name)))
        thread_lst.append(LibThread(cmd))

    # start and wait all threads
    for itr in thread_lst:
        itr.start()
    for itr in thread_lst:
        itr.join()
