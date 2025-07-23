import math
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence


SO = 100
K = 105
T = 1.0
r = 0.05
sigma = 0.2

I = 100000
rs = RandomState(MT19937(SeedSequence(I)))

z = rs.standard_normal(I)

ST = SO * np.exp((r - sigma ** 2/2) * T + sigma * math.sqrt(T) * z)
hT = np.maximum(ST - K, 0)
CO = math.exp(-r * T) * np.mean(hT)
print('Value of the European call option: {:5.3f}'.format(CO))