# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi
from ssa import SSA

def gauss_2d(X, Y, mu, sigma, ):
    return np.exp(-(
          0.5/sigma[0]**2 * (X - mu[0])**2 
        + 0.5/sigma[1]**2 * (Y - mu[1])**2))

nx = 100
ny = nx

x = np.linspace(-5., 5., nx)
y = np.linspace(-5., 5., ny)
X, Y = np.meshgrid(x, y)

trend = gauss_2d(X, Y, (0.,0.), (0.95, 0.95))

T1, T2 = 1.0, 5.0
f1, f2 = 2. * pi / T1, 2. * pi / T2
p1, p2 = 0.1, 0.05
periodic1 = p1 * np.sin(f1 * (X + Y))
periodic2 = p2 * np.sin(f2 * (X + Y))

np.random.seed(123) 
noise = 0.01 * np.random.rand(x.size, y.size)

F = trend + periodic1 + periodic2 + noise

lx = 20
ly = 20

f_ssa = SSA(F, (lx, ly))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, F, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# customize the z axis
from matplotlib.ticker import LinearLocator, FormatStrFormatter
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


F_reconstruct = f_ssa.reconstruct_elementary(range(10))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, F_reconstruct, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# customize the z axis
from matplotlib.ticker import LinearLocator, FormatStrFormatter
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, F-F_reconstruct, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# customize the z axis
from matplotlib.ticker import LinearLocator, FormatStrFormatter
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()