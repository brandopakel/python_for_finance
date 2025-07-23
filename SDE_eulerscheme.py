import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import scipy.stats as scs

npr = RandomState(MT19937(SeedSequence(100)))

S0 = 100 # initial index level
r = 0.05 # constant riskless short rate
sigma = 0.25 # constant volatility factor
I = 10000 # number of paths to be simulated
M = 50 # number of time intervals for the discretization
T = 2.0 # horizon in year fractions
dt = T / M # Length of the time interval in year fractions
S = np.zeros((M+1, I)) # 2-d ndarray object for the index levels
S[0] = S0 # inital values for the inital point in time t = 0

ST1 = S0 * np.exp((r-0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * npr.standard_normal(I)) # simulation via a vectorized expression; the discretization scheme makes use of the standard_normal() function

ST2 = S0 * npr.lognormal((r-0.5 * sigma ** 2) * T, sigma*math.sqrt(T), size=I) # simulation via a vectorized expression; the discretization scheme makes use of the lognormal() function

for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * npr.standard_normal(I)) # simulation via semivectorized expression; the loop is over the points in time starting at t=1 and ending at t=T

def print_statistics(a1, a2):
    """
    Parameters
    ==========
    a1, a2: ndarray objects
        results objects from simulation
    """

    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)

    print('%14s %14s %14s' % ('statistic', 'data set 1', 'data set 2'))
    print(45 * "-")
    print('%14s %14.3f %14.3f' % ('size', sta1[0], sta2[0]))
    print('%14s %14.3f %14.3f' % ('min', sta1[1][0], sta2[1][0]))
    print('%14s %14.3f %14.3f' % ('max', sta1[1][1], sta2[1][1]))
    print('%14s %14.3f %14.3f' % ('mean', sta1[2], sta2[2]))
    print('%14s %14.3f %14.3f' % ('std', np.sqrt(sta1[3]), np.sqrt(sta2[3])))
    print('%14s %14.3f %14.3f' % ('skew', sta1[4], sta2[4]))
    print('%14s %14.3f %14.3f' % ('kurtosis', sta1[5], sta2[5]))

print_statistics(S[-1], ST2)


plt.figure(figsize=(10,6))
plt.plot(S[:, :10], lw=1.5) # S[-1] visualization of the distribution from the final outcome (terminal) # S[:, :10] 10 simulated paths; dynamic visualization of the geometric brownian motion paths
plt.xlabel('time')
plt.ylabel('index level')
plt.show()