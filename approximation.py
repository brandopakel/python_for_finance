import numpy as np
from pylab import plt
import matplotlib as mpl

def f(x):
    return np.sin(x) + 0.5 * x

def create_plot(x, y, styles, labels, axlabels):
    plt.figure(figsize=(10,6))
    for i in range(len(x)):
        plt.plot(x[i],y[i], styles[i], label=labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
        plt.legend(loc=0)

x = np.linspace(-2 * np.pi, 2*np.pi, 50)
#create_plot([x], [f(x)], ['b'],['f(x)'],['x','f(x)'])
#plt.show()

#polyfit() for determining optimal paramters
res = np.polynomial.polynomial.polyfit(x, f(x), deg=1, full=True)
#polyval() for evaluating the approximation given a set of input values
ry = np.polynomial.polynomial.polyval(x, res[0])
#create_plot([x,x],[f(x),ry],['b','r'],['f(x)','regression'],['x','f(x)'])
#plt.show()

#higher order monomials - up to order of 5
reg = np.polynomial.polynomial.polyfit(x, f(x), deg=5)
ry = np.polynomial.polynomial.polyval(x, reg)
#create_plot([x,x],[f(x),ry],['b','r'],['f(x)','regression'],['x','f(x)'])
#plt.show()

#higher order - order of 7
reg = np.polynomial.polynomial.polyfit(x, f(x), deg=7)
ry = np.polynomial.polynomial.polyval(x, reg)
print(np.allclose(f(x), ry))
print(np.mean((f(x)-ry) ** 2)) #checks Mean Squared Error (MSE)
#create_plot([x,x],[f(x),ry],['b','r'],['f(x)','regression'],['x','f(x)'])
#plt.show()