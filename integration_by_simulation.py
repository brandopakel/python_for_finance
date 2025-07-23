import scipy.integrate as sci
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from numpy.random import MT19937, RandomState, SeedSequence

def f(x):
    return np.sin(x) + 0.5 * x

x = np.linspace(0,10)
y = f(x)
a = 0.5 # left integration limit
b = 9.5 # right integration limit
Ix = np.linspace(a,b) # integration interval values
Iy = f(Ix) # integration function values

for i in range(1,20):
    rs = RandomState(MT19937(SeedSequence(123456789)))
    x = rs.random(i*10) * (b-a) + a
    print(np.mean(f(x)) * (b-a))