import scipy.integrate as sci
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x) + 0.5 * x

x = np.linspace(0,10)
y = f(x)
a = 0.5 # left integration limit
b = 9.5 # right integration limit
Ix = np.linspace(a,b) # integration interval values
Iy = f(Ix) # integration function values

print(sci.fixed_quad(f,a,b)[0]) # fixed gaussian quadrature
print(sci.quad(f,a,b)[0]) # adaptive quadrature
#print(sci.romberg(f,a,b))

xi = np.linspace(0.5,9.5,25)
print(sci.trapezoid(f(xi),xi)) # trapezoidal rule
print(sci.simpson(f(xi), xi)) # simpson's rule