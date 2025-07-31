import math
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pylab as mpl
from scipy_describe_function import print_statistics
from gen_paths_brownian import gen_paths

def normality_tests(arr):
    """Tests for normality distribution of given data set

    Parameters
    ==========
    array: ndarray
        object to generarte statistics on
    """

    print('Skew of data set %14.3f' % scs.skew(arr))
    print('Skew test p-value %14.3f' % scs.skewtest(arr)[1])
    print('Kurt of data set %14.3f' % scs.kurtosis(arr))
    print('Kurt test p-value %14.3f' % scs.kurtosistest(arr)[1])
    print('Norm test p-value %14.3f' % scs.normaltest(arr)[1])