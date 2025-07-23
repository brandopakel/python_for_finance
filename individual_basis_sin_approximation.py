import numpy as np
import pylab as plt
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

matrix = np.zeros((4, len(x)))
matrix[3,:] = np.sin(x) #new basis function exploiting knowledge about the example function
matrix[2,:] = x ** 2
matrix[1,:] = x
matrix[0,:] = 1

reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0]
ry = np.dot(reg, matrix)

print(np.allclose(f(x),ry))
print(np.mean((f(x) - ry) ** 2))

create_plot([x,x],[f(x),ry],['b','r'],['f(x)','regression'],['x','f(x)'])
plt.show()