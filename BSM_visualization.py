import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
#import numpy.random as npr
import matplotlib.pyplot as plt
import math
import scipy.stats as scs

npr = RandomState(MT19937(SeedSequence(100)))

S0 = 100 # initial index level
r = 0.05 # constant riskless short rate
sigma = 0.25 # constant volatility factor
T = 2.0 # horizontal 
I = 10000 # number of simulations
ST1 = S0 * np.exp((r-0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * npr.standard_normal(I))

ST2 = S0 * npr.lognormal((r-0.5 * sigma ** 2) * T, sigma*math.sqrt(T), size=I)

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

print_statistics(ST1, ST2)

plt.figure(figsize=(10,6))
plt.hist(ST2, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.show()