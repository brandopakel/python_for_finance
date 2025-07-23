import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import scipy.stats as scs

npr = RandomState(MT19937(SeedSequence(100)))

S0 = 100.
r = 0.05
v0 = 0.1 # initial instantaneous volatility value
kappa = 3.0
theta = 0.25
sigma = 0.1
rho = 0.6 # fixed correlation between the two brownian motions
T = 1.0

corr_mat = np.zeros((2,2))
corr_mat[0, :] = [1.0, rho]
corr_mat[1, :] = [rho, 1.0]
cho_mat = np.linalg.cholesky(corr_mat) # cholesky decomposition and resulting matrix

M = 50
I = 100000
dt = T/M

ran_num = npr.standard_normal((2, M+1, I)) # 3-d random number data set

v = np.zeros_like(ran_num[0])
vh = np.zeros_like(v)

v[0] = v0
vh[0] = v0

for t in range(1, M+1):
    ran = np.dot(cho_mat, ran_num[:, t,:]) # relevant random number subset and transforms it via the cholesky matrix
    vh[t] = (vh[t-1] + kappa * (theta - np.maximum(vh[t-1],0)) * dt + sigma * np.sqrt(np.maximum(vh[t-1],0)) * math.sqrt(dt) * ran[1]) # simulates paths based on an euler scheme

v = np.maximum(vh, 0)

S = np.zeros_like(ran_num[0])
S[0] = S0
for t in range(1, M+1):
    ran = np.dot(cho_mat, ran_num[:, t, :])
    S[t] = S[t-1] * np.exp((r-0.5 * v[t]) * dt + np.sqrt(v[t]) * ran[0] * np.sqrt(dt))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
ax1.plot(S[:, :10], lw=1.5)
#ax1.set_xlabel('index level')
ax1.set_ylabel('index level')
ax2.plot(v[:, :10], lw=1.5)
ax2.set_xlabel('time')
ax2.set_ylabel('volatility')
plt.show()