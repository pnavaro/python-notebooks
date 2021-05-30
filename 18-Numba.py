# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     comment_magics: false
#     formats: py:light,notebooks//ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Numba

# + {"slideshow": {"slide_type": "skip"}}
import numpy as np


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# <img src="https://cdn.dribbble.com/users/915978/screenshots/3034118/numba_1x.jpg" alt="Drawing" style="width: 40%;"/>
#
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - Numba is a compiler for Python array and numerical functions.
# - Numba generates optimized machine code from pure Python code with a few simple annotations
# - Python code is just-in-time optimized to performance similar as C, C++ and Fortran, without having to switch languages or Python interpreters.
# - The code is generated on-the-fly for CPU (default) or GPU hardware.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Python decorator
#
# A decorator is used to modify a function or a class. A reference to a function "func" or a class "C" is passed to a decorator and the decorator returns a modified function or class. The modified functions or classes usually contain calls to the original function "func" or class "C". 

# + {"slideshow": {"slide_type": "fragment"}}
def timeit(function):
    def wrapper(*args, **kargs):
        import time
        t1 = time.time()
        result = function(*args, **kargs)
        t2 = time.time()
        print("execution time", t2-t1)
        return result
    return wrapper

@timeit
def f(a, b):
    return a + b

print(f(1, 2))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## First example

# + {"slideshow": {"slide_type": "fragment"}}
from numba import jit
@jit
def sum(a, b):
    return a + b


# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - Compilation will be deferred until the first function execution. 
# - Numba will infer the argument types at call time.

# + {"slideshow": {"slide_type": "fragment"}}
sum(1, 2), sum(1j, 2)

# + {"slideshow": {"slide_type": "slide"}}
x = np.random.rand(10)
y = np.random.rand(10)
sum(x, y)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Performance

# + {"slideshow": {"slide_type": "fragment"}}
x = np.random.rand(10000000)

# + {"slideshow": {"slide_type": "fragment"}}
%timeit x.sum() # Numpy


# + {"slideshow": {"slide_type": "fragment"}}
@jit
def numba_sum(x):
    res= 0
    for i in range(x.size):
        res += x[i]
    return res


# + {"slideshow": {"slide_type": "fragment"}}
%timeit numba_sum(x)


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Numba methods

# + {"slideshow": {"slide_type": "fragment"}}
@jit
def jit_sum(a, b):
    return a + b


# + {"slideshow": {"slide_type": "fragment"}}
jit_sum.inspect_types() # jit_sum has not been compiled

# + {"slideshow": {"slide_type": "slide"}}
jit_sum(1, 2) # call it once with ints
jit_sum.inspect_types()

# + {"slideshow": {"slide_type": "slide"}}
jit_sum(1., 2.) # call it once with doubles
jit_sum.inspect_types()

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - `jit_sum.inspect_llvm()` returns a dict with llvm representation.
#
# LLVM is a library that is used to construct, optimize and produce intermediate and/or binary machine code.
#
# - `jit_sum.inspect_asm()` returns a dict with assembler information. 

# + {"slideshow": {"slide_type": "fragment"}}
jit_sum.py_func(1, 2) # call origin python function without numba process


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Types coercion
#
# Tell Numba the function signature you are expecting.

# + {"slideshow": {"slide_type": "fragment"}}
@jit(['int32[:](int32[:], int32[:])','int32(int32, int32)'])
def product(a, b):
    return a*b


# + {"slideshow": {"slide_type": "fragment"}}
product(2, 3), product(2.2, 3.2)

# + {"slideshow": {"slide_type": "fragment"}}
a = np.arange(10, dtype=np.int32)
b = np.arange(10, dtype=np.int32)
product(a, b)

# + {"slideshow": {"slide_type": "slide"}}
a = np.random.random(10) # Numpy arrays contain double by default
b = np.random.random(10)
try:
    product(a, b)
except TypeError as e:
    print("TypeError:",e)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Numba types
# ```C
# void,
# intp, uintp,
# intc, uintc,
# int8, uint8, int16, uint16, int32, uint32, int64, uint64,
# float32, float64,
# complex64, complex128.
# ```
# ### Arrays
# ```C
# float32[:] 
# float64[:, :]
# ```

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Numba compilation options
#
# - ** nopython ** : Compilation fails if you use pure Python objects.
# - ** nogil ** : release Python’s global interpreter lock (GIL).
# - ** cache ** : Do not recompile the function each time you invoke a Python program.
# - ** parallel ** : experimental feature that automatically parallelizes must be used in conjunction with nopython=True:
#
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Inlining
#
# Numba-compiled functions can call other compiled functions. The function calls may even be inlined in the native code, depending on optimizer heuristics.

# + {"slideshow": {"slide_type": "fragment"}}
import math
from numba import njit

@njit
def square(x):
    return x ** 2

@njit
def hypot(x, y):
    return math.sqrt(square(x) + square(y)) # square function is inlined


# + {"slideshow": {"slide_type": "fragment"}}
hypot(2., 3.)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## @vectorize decorator
#
# - Numba’s vectorize allows Python functions taking scalar input arguments to be used as NumPy ufuncs. 
# - Write your function as operating over input scalars, rather than arrays. Numba will generate the surrounding loop (or kernel) allowing efficient iteration over the actual inputs.
#
# ### Two modes of operation:
#
# 1. Eager mode: If you pass one or more type signatures to the decorator, you will be building a Numpy universal function (ufunc). 
# 2. Call-time mode: When not given any signatures, the decorator will give you a Numba dynamic universal function (DUFunc) that dynamically compiles a new kernel when called with a previously unsupported input type. 
#
#

# + {"slideshow": {"slide_type": "slide"}}
from numba import vectorize, float64, float32, int32, int64

@vectorize([float64(float64, float64)])
def f(x, y):
    return x + y


# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# If you pass several signatures, beware that you have to pass most specific signatures before least specific ones (e.g., single-precision floats before double-precision floats)

# + {"slideshow": {"slide_type": "fragment"}}
@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)])
def f(x, y):
    return x + y


# + {"slideshow": {"slide_type": "slide"}}
a = np.arange(6)
f(a, a)

# + {"slideshow": {"slide_type": "fragment"}}
a = np.linspace(0, 1, 6)
f(a, a)

# + {"slideshow": {"slide_type": "slide"}}
a = np.linspace(0, 1+1j, 6)
f(a, a)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Why not using a simple iteration loop using the @jit decorator? 
#
# The answer is that NumPy ufuncs automatically get other features such as reduction, accumulation or broadcasting.

# + {"slideshow": {"slide_type": "fragment"}}
a = np.arange(12).reshape(3, 4)
a

# + {"slideshow": {"slide_type": "fragment"}}
f.reduce(a, axis=0)

# + {"slideshow": {"slide_type": "fragment"}}
f.reduce(a, axis=1)

# + {"slideshow": {"slide_type": "slide"}}
f.accumulate(a)

# + {"slideshow": {"slide_type": "fragment"}}
f.accumulate(a, axis=1)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## The vectorize() decorator supports multiple ufunc targets:
#
# - **cpu**  *Single-threaded CPU* : small data sizes (approx. less than 1KB), no overhead.
# - **parallel** *Multi-core CPU* : medium data sizes (approx. less than 1MB), small overhead.
# - **cuda**  *CUDA GPU* big data sizes (approx. greater than 1MB), significant overhead.
#
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## The @guvectorize decorator

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - It allows you to write ufuncs that will work on an arbitrary number of elements of input arrays, and take and return arrays of differing dimensions.

# + {"slideshow": {"slide_type": "fragment"}}
from numba import guvectorize
@guvectorize([(int64[:], int64[:], int64[:])], '(n),()->(n)')
def g(x, y, res):
    for i in range(x.shape[0]):
        res[i] = x[i] + y[0]  # adds the scalar y to all elements of x


# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# This decorator has two arguments:
# - the declaration (n),()->(n) tells NumPy that the function takes a n-element one-dimension array, a scalar (symbolically denoted by the empty tuple ()) and returns a n-element one-dimension array;
# - the list of supported concrete signatures as in @vectorize; here we only support int64 arrays.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Automatic parallelization with @jit
#
# - Setting the parallel option for jit() enables this experimental Numba feature.
# - **Array Expressions like  element-wise or point-wise array operations are supported.**
#     - unary operators: + - ~
#     - binary operators: + - * / /? % | >> ^ << & ** //
#     - comparison operators: == != < <= > >=
#     - Numpy ufuncs that are supported in nopython mode.
#     - Numpy reduction functions sum and prod.
#
# - Numpy array creation functions zeros, ones, and several random functions (rand, randn, ranf, random_sample, sample, random, standard_normal, chisquare, weibull, power, geometric, exponential, poisson, rayleigh, normal, uniform, beta, binomial, f, gamma, lognormal, laplace, randint, triangular).
#
# Numpy dot function between a matrix and a vector, or two vectors. In all other cases, Numba’s default implementation is used.
#
# Multi-dimensional arrays are also supported for the above operations when operands have matching dimension and size. The full semantics of Numpy broadcast between arrays with mixed dimensionality or size is not supported, nor is the reduction across a selected dimension.
#
# http://numba.pydata.org/numba-doc/latest/user/parallel.html

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Explicit Parallel Loops
#
# Another experimental feature of this module is support for explicit parallel loops. One can use Numba’s prange instead of range to specify that a loop can be parallelized. The user is required to make sure that the loop does not have cross iteration dependencies except the supported reductions. Currently, reductions on scalar values are supported and are inferred from in-place operations. The example below demonstrates a parallel loop with a reduction (A is a one-dimensional Numpy array):

# + {"slideshow": {"slide_type": "fragment"}}
from numba import njit, prange
@njit(parallel=True)
def prange_test(A):
    s = 0
    for i in prange(A.shape[0]):
        s += A[i]
    return s


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Exercise 
# - Optimize the Laplace equation solver with numba.
#     1. Use only @jit 
#     2. Try to use @jit(nopython=True) option
#     3. Optimize the laplace function with the right signature.
#     4. Try to parallelize.

# + {"slideshow": {"slide_type": "slide"}}
%%time
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
import itertools
from numba import jit, float64
# Boundary conditions
Tnorth, Tsouth, Twest, Teast = 100, 20, 50, 50

# Set meshgrid
n, l = 64, 1.0
X, Y = np.meshgrid(np.linspace(0,l,n), np.linspace(0,l,n))
T = np.zeros((n,n))

# Set Boundary condition
T[n-1:, :] = Tnorth
T[:1, :] = Tsouth
T[:, n-1:] = Teast
T[:, :1] = Twest

def laplace(T, n):
    residual = 0.0
    for i in range(1, n-1):
        for j in range(1, n-1):
            T_old = T[i,j]
            T[i, j] = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])
            if T[i,j]>0:
                residual=max(residual,abs((T_old-T[i,j])/T[i,j]))
    return residual

residual = 1.0   
istep = 0
while residual > 1e-5 :
    istep += 1
    residual = laplace(T, n)
    print ((istep, residual), end="\r")

print("\n iterations = ",istep)
plt.rcParams['figure.figsize'] = (10,6.67)
plt.title("Temperature")
plt.contourf(X, Y, T)
plt.colorbar()
# -

# ## Vectorize performance

# + {"slideshow": {"slide_type": "slide"}}
import socket
import numpy as np
from numba import vectorize

@vectorize(['float64(float64, float64)'], target="cpu", cache=True, nopython=True)
def cpu_add(a, b):
   return a + b

@vectorize(['float64(float64, float64)'], target="parallel", cache=True, nopython=True)
def parallel_add(a, b):
   return a + b

if socket.gethostname() == "gpu-irmar.insa-rennes.fr":
    @vectorize(['float64(float64, float64)'], target="cuda", cache=True, nopython=True)
    def parallel_add(a, b):
       return a + b


# +
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import progressbar
Nrange = (2 ** np.arange(6, 12)).astype(int)

t_numpy = []
t_numba_cpu = []
t_numba_parallel = []

bar = progressbar.ProgressBar()

for N in bar(Nrange):
    # Initialize arrays

    A = np.ones(N*N, dtype=np.float32).reshape(N,N)
    B = np.ones(A.shape, dtype=A.dtype)
    C = np.empty_like(A, dtype=A.dtype)

    t1 = %timeit -oq C = A + B
    t2 = %timeit -oq C = cpu_add(A, B)
    t3 = %timeit -oq C = parallel_add(A, B)
        
    t_numpy.append(t1.best)
    t_numba_cpu.append(t2.best)
    t_numba_parallel.append(t3.best)
   
plt.loglog(Nrange, t_numpy, label='numpy')
plt.loglog(Nrange, t_numba_cpu, label='numba cpu')
plt.loglog(Nrange, t_numba_parallel, label='numba parallel')
plt.legend(loc='lower right')
plt.xlabel('Number of points')
plt.ylabel('Execution Time (s)');

# + [markdown] {"slideshow": {"slide_type": "skip"}}
# ## References
#
# * [Numba by Loic Gouarin](https://github.com/gouarin/cours_numba_2017)
# * [Numba Documentation](http://numba.pydata.org/numba-doc/latest/index.html)
# * [Numbapro](https://github.com/ContinuumIO/numbapro-examples/)
# * [Numba examples](https://github.com/harrism/numba_examples)
#
