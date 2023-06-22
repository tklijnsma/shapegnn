import os, argparse, time, sys, re, logging
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager


def setup_logger(name='demognn'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                '\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
logger = setup_logger()


def debug(flag=True):
    if flag:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def set_matplotlib_fontsizes(small=18, medium=22, large=26):
    import matplotlib.pyplot as plt
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize
    plt.rc('figure', titlesize=large)   # fontsize of the figure title

set_matplotlib_fontsizes()


def pull_arg(*args, **kwargs):
    """
    Reads a specific argument out of sys.argv, and then
    deletes that argument from sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    return args


@contextmanager
def timeit(msg):
    """
    Prints duraction of a block of code in the terminal.
    """
    try:
        logger.info(msg)
        sys.stdout.flush()
        t0 = time.time()
        yield None
    finally:
        t1 = time.time()
        logger.info(f'Done {msg[0].lower() + msg[1:]}, took {t1-t0:.2f} secs')


class Scripter:
    """
    Command line utility.

    Add `@scripter` above a function to turn it into a command line argument.
    """
    def __init__(self):
        self.scripts = {}

    def __call__(self, fn):
        """
        Stores a command line script with its name as the key.
        """
        self.scripts[fn.__name__] = fn
        return fn

    def run(self):
        script = pull_arg('script', choices=list(self.scripts.keys())).script
        logger.info(f'Running {script}')
        self.scripts[script]()


@contextmanager
def quick_ax(figsize=(10,10), outfile='tmp.png'):
    """
    Context manager to open a matplotlib Axes.
    Upon closing, saves it to a file and calls 
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        yield ax
    finally:
        plt.savefig(outfile, bbox_inches='tight')
        try:
            os.system(f'imgcat {outfile}')
        except:
            pass

@contextmanager
def quick_fig(figsize=(10,10), outfile='tmp.png'):
    """
    Context manager to open a matplotlib Figure.
    Upon closing, saves it to a file and calls 
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        yield fig
    finally:
        plt.savefig(outfile, bbox_inches='tight')
        try:
            os.system(f'imgcat {outfile}')
        except:
            pass

