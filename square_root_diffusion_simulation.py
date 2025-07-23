import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import scipy.stats as scs

npr = RandomState(MT19937(SeedSequence(100)))

x0 = 0.05 # initial value (short rate)
kappa = 3.0 # mean reversion factor
theta = 0.02 # long-term mean value
sigma = 0.1 # volatility factor
I = 10000 # number of paths to be simulated
M = 50 # number of time intervals for the discretization
T = 2.0 # horizon in year fractions
dt = T / M # length of the time interval in year fractions

def srd_euler():
    xh = np.zeros((M + 1, I))
    x = np.zeros_like(xh)
    xh[0] = x0
    x[0] = x0
    for t in range(1, M+1):
        xh[t] = (xh[t-1] + kappa * (theta - np.maximum(xh[t-1], 0)) * dt + sigma * np.sqrt(np.maximum(xh[t-1], 0)) * math.sqrt(dt) * npr.standard_normal(I)) # simulation based on euler scheme
    x = np.maximum(xh, 0)
    return x

def srd_exact():
    x = np.zeros((M + 1, I))
    x[0] = x0
    for t in range(1, M+1):
        df = 4 * theta * kappa / sigma ** 2
        c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
        nc = np.exp(-kappa * dt) / c * x[t - 1]
        x[t] = c * npr.noncentral_chisquare(df, nc, size=I)
    return x

x1 = srd_euler()
x2 = srd_exact()

plt.figure(figsize=(10,6))
plt.plot(x2[:,:10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.show()