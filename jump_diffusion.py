import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import scipy.stats as scs

npr = RandomState(MT19937(SeedSequence(100)))

S0 = 100.
r = 0.05
sigma = 0.2
lamb = 0.75 # jump intensity
mu = -0.6 # mean jump size
delta = 0.25 # jump volatility
rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1) # drift correction

T = 1.0
M = 50
I = 10000
dt = T / M

S = np.zeros((M+1, I))
S[0] = S0
sn1 = npr.standard_normal((M+1,I))
sn2 = npr.standard_normal((M+1,I))
poi = npr.poisson(lamb * dt, (M+1, I))
for t in range(1, M+1, 1):
    S[t] = S[t-1] * (
        np.exp((r-rj-0.5*sigma**2) * dt + sigma * math.sqrt(dt) * sn1[t]) + 
        (np.exp(mu + delta * sn2[t]) - 1) * poi[t]
        )
    S[t] = np.maximum(S[t], 0)

plt.figure(figsize=(10,6))
plt.plot(S[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.show() # 2 peaks (bimodal frequency distribution) due to the jumps