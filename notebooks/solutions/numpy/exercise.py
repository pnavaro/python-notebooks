import numpy as np

print(np.arange(100,110))
print()
print(np.linspace(-2,2,20,endpoint=False))
print()
print(np.logspace(-3,-2,10))
print()
print(np.tri(7,5, k=1) - np.ones((7,5)))
print()
print(np.zeros((7,5))-np.tri(5, 7, k=-2).transpose())
