import scipy.stats as scs
import numpy as np

def print_statistics(array):
    """Print selected statistics.
    
    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    """

    sta = scs.describe(array)
    print('%14s %15s' % ('statistic','value'))
    print(30 * '-')
    print('%14s %15.5f' % ('size', sta[0]))
    print('%14s %15.5f' % ('min', sta[1][0]))
    print('%14s %15.5f' % ('max', sta[1][1]))
    print('%14s %15.5f' % ('mean', sta[2]))
    print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
    print('%14s %15.5f' % ('skew', sta[4]))
    print('%14s %15.5f' % ('kurtosis', sta[5]))