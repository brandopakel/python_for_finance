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

fig = plt.figure(figsize=(10,6)) #type: ignore
ax = fig.add_subplot()
plt.plot(x, y, 'b', linewidth = 2)
plt.ylim(bottom=0)
verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b,0)]
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)
plt.text(0.75 * (a+b), 1.5, r"$\int_a^b f(x)dx$", horizontalalignment='center', fontsize=20)
plt.figtext(0.9, 0.075, '$x$')
plt.figtext(0.075, 0.9, '$f(x)$')
ax.set_xticks((a,b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([f(a), f(b)])

plt.show()