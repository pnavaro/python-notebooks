import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,10)
x = np.linspace(0,2*np.pi, 101)
s = np.sin(x)
c = np.cos(x)
img = plt.imread('delicate_arch.png')
plt.subplot(2,2,1)
plt.plot(x,s,'b-',x,c,'r+')
plt.axis('tight') # Stretch the boundaries to match with data
plt.subplot(2,2,2)
plt.plot(x,s);
plt.grid()
plt.xlabel('radians')
plt.xlabel('amplitude')
plt.title('sin(x)')
plt.axis('tight')
plt.subplot(2,2,3)
plt.imshow(img,  cmap=plt.cm.winter)
plt.tight_layout()
plt.savefig('my_plots.png')
