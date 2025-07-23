import math
import scipy.optimize as sco
import numpy as np

def Eu(p):
    s, b = p
    return -(0.5 * math.sqrt(s * 15 + b * 5) + 0.5 * math.sqrt(s * 5 + b * 12)) # function to be minimized in order to maximize expected utility

cons = ({
    'type' : 'ineq',
    'fun': lambda p : 100 - p[0] * 10 - p[1] * 10
}) #inequality constraint as a dict object

bnds = ((0,1000), (0,1000)) # boundary values for the parameters

result = sco.minimize(Eu, [5,5], method='SLSQP', bounds=bnds, constraints=cons) # constrained optimization

print(result)

print(result['x']) # optimal parameter values (optimal portfolio)
print(-result['fun']) #type: ignore # negative minimum function value as the optimal solution value
print(np.dot(result['x'], [10,10])) # budgest constraint is binding, all wealth is invested