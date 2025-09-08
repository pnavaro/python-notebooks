import numpy as np
import itertools
import matplotlib.pyplot as plt

# Boundary conditions
Tnorth, Tsouth, Twest, Teast = 100, 20, 50, 50

# Set meshgrid
n, l = 64, 1.0
X, Y = np.meshgrid(np.linspace(0,l,n), np.linspace(0,l,n))
T = np.ones((n,n))

# Set Boundary condition
T[-1, :] = Tnorth
T[0, :] = Tsouth
T[:, -1] = Teast
T[:, 0] = Twest

for istep in itertools.count():
    T_old = T.copy()
    T[1:-1,1:-1] = (T[1:-1,2:]+T[2:,1:-1]+T[1:-1,:-2]+T[:-2,1:-1])*0.25
    residual = np.max(np.abs((T_old - T)/T_old))
    print ((istep, residual), end="\r")
    if residual < 1e-5: break

print()
print("iterations = ",istep)
plt.title("Temperature")
plt.contourf(X, Y, T)
plt.colorbar()
plt.show()
