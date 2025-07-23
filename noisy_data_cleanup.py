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

#x = np.linspace(-2*np.pi, 2*np.pi, 50)

xn = np.linspace(-2*np.pi, 2*np.pi, 50)
xn = xn + 0.15 * np.random.standard_normal(len(xn))
yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))

reg = np.polynomial.polynomial.polyfit(xn, yn, deg=7)
ry = np.polynomial.polynomial.polyval(xn, reg)
create_plot([xn,xn],[f(xn),ry],['b','r.'],['f(xn)','regression'],['x','f(xn)'])
plt.show()