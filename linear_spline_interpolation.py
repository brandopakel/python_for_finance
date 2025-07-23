import scipy.interpolate as spi
import numpy as np
import matplotlib.pyplot as plt

def create_plot(x, y, styles, labels, axlabels):
    plt.figure(figsize=(10,6))
    for i in range(len(x)):
        plt.plot(x[i],y[i], styles[i], label=labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
        plt.legend(loc=0)


def f(x):
    return np.sin(x) + 0.5 * x

x = np.linspace(-2 * np.pi, 2 * np.pi, 25)

ipo = spi.splrep(x, f(x), k=1)
iy = spi.splev(x, ipo)

np.allclose(f(x), iy)
create_plot([x,x],[f(x),iy],['b','ro'],['f(x)','interpolation'],['x','f(x)'])
plt.show()