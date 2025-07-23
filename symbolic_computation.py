import sympy as sy

x = sy.Symbol('x')
y = sy.Symbol('y')

a, b = sy.symbols('a b') #symbol objects for the integral limits

print(sy.solve(x ** 2 - 1)) # type: ignore
print(sy.solve(x ** 2 - 1 - 3)) #type: ignore
print(sy.solve(x ** 3 + 0.5 * x ** 2 - 1)) #type: ignore
print(sy.solve(x ** 2 + y ** 2)) #type: ignore

I = sy.Integral(sy.sin(x) + 0.5 * x, (x,a,b)) #type: ignore  - The Integral object defined and pretty-printed
print(sy.pretty(I))

int_func = sy.integrate(sy.sin(x) + 0.5 * x, x) #type: ignore - The antiderivative derived and pretty-printed
print(sy.pretty(int_func))

Fb = int_func.subs(x, 9.5).evalf() #type: ignore - The values of the antiderivative at the limits, obtained via the .subs() and .evalsf() methods 
Fa = int_func.subs(x, 0.5).evalf() #type: ignore -  
print(Fb - Fa) #type: ignore - The exact numerical value of the integral

int_func_limits = sy.integrate(sy.sin(x) + 0.5 * x, (x,a,b)) # type: ignore
print(sy.pretty(int_func_limits))

print(int_func_limits.subs({
    a: 0.5,
    b: 9.5
}).evalf()) #type: ignore

print(int_func.diff()) #type:ignore - Derivative of the antiderivative yields the original function


f = (sy.sin(x) + 0.05 * x ** 2 + sy.sin(y) + 0.05 * y ** 2) #type: ignore - The symbolic version of the function
del_x = sy.diff(f, x) # The two partial derivatives derived and printed
del_y = sy.diff(f, y)
xo = sy.nsolve(del_x, -1.5) # Educated guesses for the roots and resulting optimal values
yo = sy.nsolve(del_y, -1.5)
print(f.subs({
    x: xo,
    y: yo
}).evalf()) # The global minimum function value