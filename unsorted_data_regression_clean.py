import numpy as np
import pylab as plt

def f(x):
    return np.sin(x) + 0.5 * x

def create_plot(x, y, styles, labels, axlabels):
    plt.figure(figsize=(10,6))
    for i in range(len(x)):
        plt.plot(x[i],y[i], styles[i], label=labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
        plt.legend(loc=0)

xu = np.random.random_sample(50) * 4 * np.pi - 2 * np.pi
yu = f(xu)

reg = np.polynomial.polynomial.polyfit(xu, yu, deg=5)
ry = np.polynomial.polynomial.polyval(xu, reg)
create_plot([xu,xu],[yu,ry],['b.','ro'],['f(x)','regression'],['x','f(x)'])
plt.show()