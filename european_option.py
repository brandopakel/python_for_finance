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

def gbm_mcs_stat(K):
    """ Valuation of European call option in Black-Scholes-Merton by Monte Carlo Simulation (of index level at maturity)

    Parameters
    ==========
    K: float
        (positive) strike price of the option
    
    Returns
    ========
    C0: float
        estimated present value of European call option
    """

    sn = gen_sn(1, I)
    # simulate index level at maturity
    ST = S0 * np.exp((r-0.5 * sigma**2) * T + sigma * math.sqrt(T) * sn[1])
    # calculate payoff at maturity
    hT = np.maximum(ST - K, 0)
    # calculate MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0

def gbm_mcs_dyn(K, option='call'):
    """ Valuation of European call option in Black-Scholes-Merton by Monte Carlo Simulation (of index level at maturity)

    Parameters
    ==========
    K: float
        (positive) strike price of the option
    option : string
        type of the option to be valued ('call', 'put') 
    
    Returns
    ========
    C0: float
        estimated present value of European call option
    """

    dt = T/M
    # simulation of index level paths
    S = np.zeros((M+1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M+1):
        S[t] = S[t-1] * np.exp((r-0.5*sigma**2) * dt + sigma * math.sqrt(dt) * sn[t])
    # case-based calculation of payoff
    if option == 'call':
        hT = np.maximum(S[-1] - K, 0)
    else:
        hT = np.maximum(K - S[-1], 0)
    # calculate MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0

print(gbm_mcs_dyn(K=110., option='put'))