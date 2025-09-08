import numpy as np
s = 5
a = -np.ones((s,s)) 
for i in range(s):
  a += np.diag((i+1)*np.ones(s-i), k=i)
a
# trapezoidal rule
def trapz( f , a, b, n):
    """ compute the integral of f in [a,b]
    using trapezoidal rule with n partitions """
    x, dx = np.linspace(a, b, n, retstep=True)
    return np.sum(0.5*(f(x[:-1])+f(x[1:])))*dx

f = lambda v: np.exp(-v*v)

print(trapz(f, -10, 10, 20))
x = np.linspace(-10, 10, 20)
print(np.trapz(f(x), x))  # check the result with numpy.trapz function 
