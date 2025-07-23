import scipy.optimize as sco
import numpy as np

# local convex optimization draws on the results of the global optimization
# for many convex optimization problems it is advisable to have a global minimization before the local one
# local convex optimizations can esasily be trapped in a local minimum ('basin hopping')

def fo(p):
    x, y = p
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    if output == True:
        print('%8.4f | %8.4f | %8.4f' % (x,y,z))
    return z

def fm(p):
    x, y = p
    return (np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2)

output = False
opt1 = sco.brute(fo, ((-10,10.1,0.1), (-10,10.1,0.1)), finish=None)

opt2 = sco.fmin(fo, opt1, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20) 

print(sco.fmin(fo, (2.0, 2.0), maxiter=250))