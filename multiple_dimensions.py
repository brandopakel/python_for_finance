import numpy as np
from matplotlib.projections import register_projection
from matplotlib.axes._subplots import SubplotBase
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as axes3d

class Axes3DSubplot(Axes3D, SubplotBase): pass
register_projection(Axes3DSubplot)
axes3d.Axes3DSubplot = Axes3DSubplot

#print(matplotlib.projections.get_projection_names())
#print(matplotlib.projections.get_projection_class('3d'))

#print("matplotlib version:", matplotlib.__version__)
#print("Matplotlib backend:", matplotlib.get_backend())

def fm(p):
    x, y = p
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#print("AX repr:", repr(ax))
#print("AX type:", type(ax))
#print("AX module:", type(ax).__module__)
#print("AX base classes:", type(ax).__bases__)
#print("AX plot_surface?", hasattr(ax, "plot_surface"))

#print("Type:", type(ax))
#print("Dir:", [method for method in dir(ax) if "surface" in method])

#print("Has plot_surface?", hasattr(ax, 'plot_surface'))
#print(repr(ax))
x = np.linspace(0, 10, 20)
y = np.linspace(0,10,20)
X, Y = np.meshgrid(x, y)

Z = fm((X,Y))
x = X.flatten() #yields 1D ndarray object from 2D ndarray object
y = Y.flatten()