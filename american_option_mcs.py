import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import scipy.stats as scs
from variance_reduction_function import gen_sn

S0 = 100.
r = 0.05
sigma = 0.25
T = 1.0
I = 50000
M = 50

def gbm_mcs_amer(K, option='call'):
    """ Valuation of American option in Black-Scholes-Merton by Monte Carlo simulation by LSM algorithm

    Parameters
    ==========
    K: float
        (positive) strike price of the option
    option: string
        type of the option to be valued ('call','put')
    
    Returns
    =======
    C0: float
        estimated present value of American call option
    """

    dt = T/M
    df = math.exp(-r * dt)
    # simulation of index levels
    S = np.zeros((M+1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M+1):
        S[t] = S[t-1] * np.exp((r-0.5 * sigma **2) * dt + sigma * math.sqrt(dt) * sn[t])
    # case based calculation of payoff
    if option == 'call':
        h = np.maximum(S-K, 0)
    else:
        h = np.maximum(K-S, 0)
    # LSM algorithm
    V = np.copy(h)
    for t in range(M-1, 0, -1):
        reg = np.polyfit(S[t], V[t+1]*df, 7)
        C = np.polyval(reg, S[t])
        V[t] = np.where(C > h[t], V[t+1]*df, h[t])
    # MCS estimator
    C0 = df * np.mean(V[1])
    return C0