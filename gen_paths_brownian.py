import math
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pylab as mpl
from scipy_describe_function import print_statistics

def gen_paths(S0, r, sigma, T, M, I):
    """Generate Monte Carlo Paths for geometric Brownian motion

    Parameters
    ==========
    S0: float
        initial stock/index value
    r: float
        constant short rate
    sigma: float
        constant volatility
    T: float
        final time horizon
    M: int
        number of time steps/intervals
    I: int
        number of paths to be simulated
    
    Returns
    =======
    paths: ndarray, shape(M+1, I)
        simulated paths given the parameters
    """

    dt = T/M
    paths = np.zeros((M+1, I))
    paths[0] = S0
    rng = np.random.default_rng()
    for t in range(1, M+1):
        rand = rng.standard_normal(I)
        rand = (rand - np.mean(rand)) / np.std(rand)
        paths[t] = paths[t-1] * np.exp((r-0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rand)

    return paths

# test
S0 = 100. # initival value for simulated processes
r = 0.05 # const short rate
sigma = 0.2 # constant volatility factor
T = 1.0 # time horizon in year fractions
M = 50 # number of time intervals
I = 250000 # number of simulated processes
paths = gen_paths(S0, r, sigma, T, M, I)
log_returns = np.log(paths[1:] / paths[:-1])

#print(paths[:, 0].round(4))
#print(np.info(paths[:,0]))
#print(S0 * math.exp(r * T))
#print(paths[-1].mean()) # expected value and average simulated value
#print(log_returns[:, 0].round(4))

#print_statistics(log_returns.flatten())

#mpl.figure(figsize=(10,6))
#mpl.plot(paths[:,:10]) # ten simulated paths of geometric brownian motion
#mpl.hist(log_returns.flatten(), bins=70, density=True, label='frequency',color='b')
#sm.qqplot(log_returns.flatten()[::500],line='s') # quantile-quantile plot for log returns of geometric brownian motion
#mpl.xlabel('theoretical quantiles')
#mpl.ylabel('sample quantiles')
#x = np.linspace(mpl.axis()[0], mpl.axis()[1])
#mpl.plot(x, scs.norm.pdf(x, loc=r / M, scale=sigma / np.sqrt(M))) # histogram of log returns of geometric brownian motion and normal density function
#mpl.show()