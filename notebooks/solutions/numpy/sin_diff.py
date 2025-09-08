import numpy as np
import matplotlib.pyplot as plt

x, dx = np.linspace(0,6*np.pi,60, retstep=True)
y = x * np.sin(x)

dy = y[1:]-y[:-1]
dx = x[1:]-x[:-1]
dy_dx = dy / dx

centers_x = 0.5*(x[1:]+x[:-1]) 
plt.plot(x, np.sin(x) + x * np.cos(x),'b', label="expected")
plt.plot(centers_x, dy_dx, 'ro', label="computed")
plt.title(r"$\frac{d(x\sin(x))}{dx}$");
plt.legend()
plt.xlabel('radians')
plt.ylabel('amplitude');
plt.show()
