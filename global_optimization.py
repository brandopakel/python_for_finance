import scipy.optimize as sco
import numpy as np

output = False

def fo(p):
    x, y = p
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    if output == True:
        print('%8.4f | %8.4f | %8.4f' % (x,y,z))
    return z

def fm(p):
    x, y = p
    return (np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2)

opt1 = sco.brute(fo, ((-10,10.1,0.1), (-10,10.1,0.1)), finish=None) #brute force optimization
print(opt1) #optimal parameters are (-1.4 - 1.4) for x and y
print(fm(opt1)) #minimal function value for the global minimization is -1.7748994