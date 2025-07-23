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

xd = np.linspace(1.0,3.0,50)

ipo = spi.splrep(xd, f(xd), k=1)
iy = spi.splev(xd, ipo)

print(np.allclose(f(xd), iy))
create_plot([xd,xd],[f(xd),iy],['b','ro'],['f(x)','interpolation'],['x','f(x)'])
plt.show()