import numpy as np
import matplotlib.pyplot as plt
# Boundary conditions
Tnorth, Tsouth, Twest, Teast = 100, 20, 50, 50

# Set meshgrid
n, l = 64, 1.0
X, Y = np.meshgrid(np.linspace(0,l,n), np.linspace(0,l,n))
T = np.zeros((n,n))

# Set Boundary condition
T[-1, :] = Tnorth
T[0, :] = Tsouth
T[:,-1:] = Teast
T[:,0] = Twest


istep = 0
residual = 1.0
while residual > 1e-5 :
    istep += 1
    print ((istep, residual), end="\r")
    T_old = np.copy(T)
    for i in range(1, n-1):
        for j in range(1, n-1):
            T[i, j] = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])

    residual=np.max(np.abs((T_old-T)/T))


print()
print("iterations = ",istep)
plt.title("Temperature")
plt.contourf(X, Y, T)
plt.colorbar()
