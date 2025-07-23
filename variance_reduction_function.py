import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import scipy.stats as scs

npr = RandomState(MT19937(SeedSequence(100)))

def gen_sn(M, I, anti_paths = True, mo_match = True):
    """
    Parameters
    =========
    M: int
        number of time intervals for discretization
    I: int
        number of paths to be simulated
    anti_paths: boolean
        use of antithetic variates
    mo_math: boolean
        use of moment matching
    """

    if anti_paths is True:
        sn = npr.standard_normal((M+1, int(I/2)))
        sn = np.concatenate((sn,-sn), axis=1)
    else:
        sn = npr.standard_normal((M+1, I))
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    return sn