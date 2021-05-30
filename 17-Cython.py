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

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# # Cython

# + {"slideshow": {"slide_type": "fragment"}}
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)
%config InlineBackend.figure_format = 'retina'
import numpy as np

# + {"slideshow": {"slide_type": "fragment"}}
import warnings
warnings.filterwarnings("ignore")


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ![Cython logo](images/440px-Cython-logo.svg.png)

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# * Cython  provides extra syntax allowing for static type declarations (remember: Python is generally dynamically typed)
# * Python code gets translated into optimised C/C++ code and compiled as Python extension modules
# * Cython allows you to write fast C code in a Python-like syntax. 
# * Furthermore, linking to existing C libraries is simplified.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - Pure Python Function
#
#
# $f(x)=-2x^3+5x^2+x$,

# + {"slideshow": {"slide_type": "fragment"}}
def f(x):
    return -4*x**3 +3*x**2 +2*x

x = np.linspace(-1,1,100)
ax = plt.subplot(1,1,1)
ax.plot(x, f(x))
ax.fill_between(x, 0, f(x));


# + [markdown] {"slideshow": {"slide_type": "slide"}}
#  we compute integral $\int_a^b f(x)dx$ numerically with $N$ points.

# + {"slideshow": {"slide_type": "fragment"}}
def integrate_f_py(a,b,N):
    s  = 0
    dx = (b - a) / (N-1)
    for i in range(N-1): # we intentionally use the bad way to do this with a loop
        x = a + i*dx
        s += (f(x)+f(x+dx))/2
    return s*dx


# + {"slideshow": {"slide_type": "fragment"}}
%timeit integrate_f_py(-1,1,10**3)
print(integrate_f_py(-1,1,1000))

# + {"slideshow": {"slide_type": "slide"}}
%load_ext heat

# + {"slideshow": {"slide_type": "fragment"}}
%%heat
def f(x):
    return -4*x**3 +3*x**2 +2*x
def integrate_f(a, b, N):
    s  = 0
    dx = (b - a) / (N-1)
    for i in range(N-1):
        x = a + i*dx
        s += (f(x)+f(x+dx))/2
    return s*dx

integrate_f(0, 10, 1000)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - Pure C function
#

# + {"slideshow": {"slide_type": "slide"}}
%%file integral_f_c.c

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define NB_RUNS 1000

double f(double x) {
    return -4*x*x*x +3*x*x +2*x;
}

double integrate_f_c(double a, double b, int N) {
    double s  = 0;
    double dx = (b - a) / (N-1);
    for(int i=0; i<N-1; ++i){
        double x = a + i*dx;
        s += (f(x)+f(x+dx))/2.0;
    }
    return s*dx;
}

int main(int argc, char **argv)
{
  
  double a =  atof(argv[1]);
  double b =  atof(argv[2]);
  int N    =  atoi(argv[3]);
  double  res = 0;

  clock_t begin = clock();

  for (int i=0; i<NB_RUNS; ++i)
      res += integrate_f_c( a, b, N );
    
  clock_t end = clock();
     
  fprintf( stdout, "integral_f(%3.1f, %3.1f, %d) = %f \n", a, b, N, res / NB_RUNS );
  fprintf( stdout, "time = %e ms \n",  (double)(end - begin) / CLOCKS_PER_SEC );

  return 0;
}


# + {"slideshow": {"slide_type": "subslide"}}
!gcc -O3 integral_f_c.c; ./a.out -1 1 1000

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Cython compilation: Generating C code
#
# Load Cython in jupyter notebook.

# + {"slideshow": {"slide_type": "fragment"}}
%load_ext Cython

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# ### C Variable and Type definitions
#
# In general, use `cdef` to declare C variables. 
# The command :
# ```sh
# $ cython -a mycode.pyx
# ```
# outputs an html file. It shows what parts of your code are C, which parts are Python, and where C-Python conversion occurs.

# + {"slideshow": {"slide_type": "slide"}, "magic_args": "-a", "language": "cython"}
# cdef int i, j = 2, k = 3      # assigning values at declaration
# i = 1                         # assigning values afterwards
# # avoid Python-C conversion! It's expensive:
# a = 5
# i = a
# # same with C-Python conversion:
# b = j
# print("a = %d" % a)
# print("i = %d" % i)
# print("b = %d" % b)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Another Python vs. Cython coloring guide

# + {"slideshow": {"slide_type": "fragment"}, "magic_args": "-a", "language": "cython"}
# cdef int m, n
# cdef double cy_total = 0.0
# for m in range(10):
#     n = 2*m
#     cy_total += n
# a, b = 0, 0
# py_total = 0.0
# for a in range(10):
#     b = 2*a
#     py_total += b
# print(cy_total, py_total)

# + {"slideshow": {"slide_type": "slide"}, "magic_args": "-a", "language": "cython"}
# cdef struct Grail:
#     int age
#     float volume
# cdef union Food:
#     char *spam
#     float *eggs
# cdef enum CheeseType:
#     cheddar, edam,
#     camembert
# cdef enum CheeseState:
#     hard = 1
#     soft = 2
#     runny = 3
# cdef Grail holy
# holy.age    = 500
# holy.volume = 10.0
# print (holy.age, holy.volume)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Cython Functions
#
# Use **cdef** to define a Cython function.   
#  - Cython function can accept either (inclusive) Python and C values as well as return either Python or C values,
#  - *Within a Cython module* Python and Cython functions can call each other freely. However, only **Python** functions can be called from outside the module by Python code. (i.e. importing/exporting a Cython module into some Python code)
#
# **cpdef** define a Cython function with a simple Python wrapper. However, when called from Cython the Cython / C code is called directly, bypassing the Python wrapper.  

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# Writing pure code in Cython gives a small speed boost. Note that none of the code below is Cython-specific. Just add `.pyx` instead of `.py` extension.

# + {"slideshow": {"slide_type": "fragment"}}
%%file cython_f_example.pyx
def f(x):
    return -4*x**3 +3*x**2 +2*x
def integrate_f(a, b, N):
    s  = 0
    dx = (b - a) / (N-1)
    for i in range(N-1):
        x = a + i*dx
        s += (f(x)+f(x+dx))/2
    return s*dx


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Cython Compilation
#
# - The .pyx source file is compiled by Cython to a .c file.
# - The .c source file contains the code of a Python extension module.
# - The .c file is compiled by a C compiler to a .so (shared object library) file which can be imported directly into a Python session.

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# ### Build with CMake
# ```cmake
# project(cython_f_example CXX)
# include(UseCython)  # Load Cython functions
# # Set C++ output
# set_source_file_properties(cython_f_example.pyx PROPERTIES CYTHON_IS_CXX TRUE )
# # Build the extension module
# cython_add_module( modname cython_f_example.pyx cython_f_example.cpp )
# ```

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### C/C++ generation with cython application
# ```sh
# cython -3 cython_f_example.pyx   # create the C file for Python 3
# cython -3 --cplus cython_f_example.pyx  # create the C++ file for Python 3
# ```

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
#
# ### build with a C/C++ compiler
# To build use the Makefile:
# ```make
# CC=gcc
# CFLAGS=`python-config --cflags` 
# LDFLAGS=`python-config --ldflags`
# cython_f_example:
# 	 ${CC} -c $@.c ${CFLAGS}
# 	 ${CC} $@.o -o $@.so -shared ${LDFLAGS}
# ```
# Import the module in Python session
# ```python
# import cython_f_example
# ```
#
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## pyximport
#
# import Cython .pyx files as if they were .py files:

# + {"slideshow": {"slide_type": "fragment"}}
import pyximport
pyximport.install()
import cython_f_example
%timeit cython_f_example.integrate_f(-1,1,10**3)
print(cython_f_example.integrate_f(-1,1,1000))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Building a Cython module using distutils
#
# Create the setup.py script:

# + {"slideshow": {"slide_type": "fragment"}}
%%file setup.py
from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Cython Example Integrate f Function',
  ext_modules = cythonize("cython_f_example.pyx"),
)

# + {"slideshow": {"slide_type": "slide"}}
%run setup.py  build_ext --inplace --quiet

# + {"slideshow": {"slide_type": "fragment"}}
from cython_f_example import integrate_f
%timeit integrate_f(-1,1,10**3)
integrate_f(-1,1,10**3)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Why is it faster with Cython ?
#
# - Python code is interpreted at every execution to machine code.
# - Compiled C code is already in machine code.
# - C is a statically-typed language. It gives to the compiler more information which allows it to optimize both computations and memory access.
# - To add two variables, Python checks the type before calling the right __add__ function and store it to a value that can be new.
# - C just add the variables and return the result.
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Add Cython types 
# We coerce Python types to C types when calling the function. Still a "Python function" so callable from the global namespace.

# + {"slideshow": {"slide_type": "fragment"}, "language": "cython"}
# def f(x):
#     return -4*x**3 +3*x**2 +2*x
# def cy_integrate_f(double a, double b, int N):
#     cdef int i
#     cdef double s, x, dx
#     s  = 0
#     dx = (b - a) / (N-1)
#     for i in range(N-1):
#         x = a + i*dx
#         s += (f(x)+f(x+dx))/2
#     return s*dx

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# * typing the iterator variable i with C semantics, tells Cython to compile the for-loop to pure C code.
# * typing a, s and dx is important as they are involved in arithmetic within the for-loop
#
# * Cython type declarations can make the source code less readable
# * Do not use them without good reason, i.e. only in performance critical sections.

# + {"slideshow": {"slide_type": "fragment"}}
%timeit cy_integrate_f(-1,1,10**3)
print(cy_integrate_f(-1,1,1000))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# Finally, we integrate a Cython function instead of a Python function. 
# This eliminates the Python-C conversion at the function call as seen 
# above thus giving a pure Cython/C algorithm.
#
# The primary downside is not being allowed to call
# the function `cy_f`, from Python unless `cpdef` is used. 

# + {"slideshow": {"slide_type": "fragment"}, "language": "cython"}
# cdef double cy_f(double x):
#     return -4*x**3 +3*x**2 +2*x
# def cycy_integrate_f(double a, double b, int N):
#     cdef int i
#     cdef double s, x, dx
#     s  = 0
#     dx = (b - a) / (N-1)
#     for i in range(N-1):
#         x = a + i*dx
#         s += (cy_f(x)+cy_f(x+dx))/2
#     return s*dx

# + {"slideshow": {"slide_type": "fragment"}}
%timeit cycy_integrate_f(-1,1,10**3)
print(cycy_integrate_f(-1,1,1000))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Exercise : Cythonize the trivial exponential function.

# + {"slideshow": {"slide_type": "fragment"}, "magic_args": "-a", "language": "cython"}
# def exp_python(x,terms=50):
#     sum = 0.
#     power = 1.
#     fact = 1.
#     for i in range(terms):
#         sum += power/fact
#         power *= x
#         fact *= i+1
#     return sum

# + {"slideshow": {"slide_type": "fragment"}}
%timeit exp_python(1.,50)

# + {"slideshow": {"slide_type": "skip"}, "language": "cython"}
#
# # %load solutions/cython/exponential.pyx
# #cython: profile=False
# #cython: cdivision=True
# def exp_cython(double x, int terms = 50):
#    cdef double sum
#    cdef double power
#    cdef double fact
#    cdef int i
#    sum = 0.
#    power = 1.
#    fact = 1.
#    for i in range(terms):
#       sum += power/fact
#       power *= x
#       fact *= i+1
#    return sum
#

# + {"slideshow": {"slide_type": "skip"}}
%timeit exp_cython(1.,50)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Cython and Numpy
#
# The Numpy library contains many fast numerics routines. Their speed comes 
# from manipulating the low-level C-arrays that the numpy.array object wraps 
# rather than computing over slow Python lists. Using Cython one can access 
# those low-level arrays and implement their own fast algorithms while allowing 
# the easy interaction afforded by Python + Numpy.
#
# The examples below are various implementations of the naive matrix multiplication 
# algorithm. We will start with a pure Python implementation and then incrementally 
# add structures that allow Cython to exploit the low-level speed of the numpy.array 
# object.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Pure Python implementation compiled in Cython without specific optimizations.

# + {"slideshow": {"slide_type": "fragment"}, "language": "cython"}
# def matmul1(A, B, out=None):
#     assert A.shape[1] == B.shape[0]
#     for i in range(A.shape[0]):
#         for j in range(B.shape[1]):
#             s = 0
#             for k in range(A.shape[1]):
#                 s += A[i,k] * B[k,j]
#             out[i,j] = s
#     return out

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Import numpy as a Cython module
#
# We now take advantage of the ability to access the underlying C arrays in the `numpy.array` object from Cython, thanks to a special `numpy.pxd` file included with Cython. (The Cython developers worked closely with Numpy developers to make this optimal.) 
#
# To begin with, we have to `cimport` numpy: that is, import numpy as a **Cython** module rather than a **Python** module. To do so, simply type:
#
# ```python
# cimport numpy as np
# ```
# Another important thing to note is the type of Numpy indexers. There is a special Numpy variable type used for `numpy.array` indices called `Py_ssize_t`. To take full advantage of the speedups that Cython can provide we should make sure to type the variables used for indexing as such.
#

# + {"slideshow": {"slide_type": "slide"}, "language": "cython"}
# import numpy as np
# cimport numpy as np
# ctypedef np.float64_t dtype_t      # shorthand type. easy to change
# def matmul2(np.ndarray[dtype_t, ndim=2] A,
#             np.ndarray[dtype_t, ndim=2] B,
#             np.ndarray[dtype_t, ndim=2] out=None):
#     cdef Py_ssize_t i, j, k
#     cdef dtype_t s
#     assert A.shape[1] == B.shape[0]
#     for i in range(A.shape[0]):
#         for j in range(B.shape[1]):
#             s = 0
#             for k in range(A.shape[1]):
#                 s += A[i,k] * B[k,j]
#             out[i,j] = s
#     return out

# + {"slideshow": {"slide_type": "slide"}}
import numpy as np
from timeit import timeit
A = np.random.random_sample((64,64))
B = np.random.random_sample((64,64))
C = np.zeros((64,64))

# + {"slideshow": {"slide_type": "fragment"}}
%timeit matmul1(A,B,C)

# + {"slideshow": {"slide_type": "fragment"}}
%timeit matmul2(A,B,C)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Tuning indexing
# The array lookups are still slowed down by two factors:
#   * Bounds checking is performed.
#   * Negative indices are checked for and handled correctly. 
#   
# The code doesn’t use negative indices, and always access to arrays within bounds. We can add a decorator to disable bounds checking:

# + {"slideshow": {"slide_type": "slide"}, "language": "cython"}
# cimport cython                                       # cython tools
# import numpy as np
# cimport numpy as np
# ctypedef np.float64_t dtype_t
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def matmul3(np.ndarray[dtype_t, ndim=2] A,
#             np.ndarray[dtype_t, ndim=2] B,
#             np.ndarray[dtype_t, ndim=2] out=None):
#     cdef Py_ssize_t i, j, k
#     cdef dtype_t s
#     assert A.shape[1] == B.shape[0]
#     for i in range(A.shape[0]):
#         for j in range(B.shape[1]):
#             s = 0
#             for k in range(A.shape[1]):
#                 s += A[i,k] * B[k,j]
#             out[i,j] = s
#     return out

# + {"slideshow": {"slide_type": "fragment"}}
%timeit matmul3(A,B,C)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Cython Build Options
#
# - boundcheck(True,False) : array bounds checking
# - wraparound(True,False) : negative indexing.
# - initializedcheck(True,False): checks that a memoryview is initialized 
# - nonecheck(True,False) : Check if one argument is  None
# - overflowcheck(True,False) : Check if int are too big
# - cdivision(True,False) : If False, adjust the remainder and quotient operators C types to match those of Python ints. Could be very effective when it is set to True.
# - profile (True / False) : Write hooks for Python profilers into the compiled C code. Default is False.
#
# [Cython Compiler directives](http://docs.cython.org/en/latest/src/reference/compilation.html#compiler-directives)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Numpy objects with external C program. 
#
# Note that this can actually be slower 
# because the C function is not the best implementation of matrix multiplication. Call cblas with same technique is an interesting exercise.

# + {"slideshow": {"slide_type": "fragment"}}
%%file mydgemm.c 
void my_dgemm( int m, int n, int k, 
              double a[m][n], double b[n][k], float c[m][k] )
{
  double ab = 0;
  for( int j = 0 ; j < m ; j++ ) {
    for( int i = 0 ; i < k ; i++ ) {
      for( int l = 0 ; l < n ; l++ ){
        ab += a[j][l] * b[l][i];
      }
      c[j][i] = ab ;
      ab = 0;
    }
  }
}

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - The `np.ndarray[double, ndim=2, mode="c"]` assures that you get a C-contiguous numpy array of doubles 
# - The `&input[0,0]` passed in the address of the beginning of the data array.
# -

from pyximport import install
import os
here = os.getcwd()

# + {"slideshow": {"slide_type": "fragment"}, "magic_args": "-I {here}", "language": "cython"}
# # do not forget to change the file path
# cdef extern from "mydgemm.c":
#     void my_dgemm (int m, int n, int k, 
#                           double *A, double *B, double *C)
# cimport cython
# import numpy as np
# cimport numpy as np
# ctypedef np.float64_t dtype_t
# @cython.boundscheck(False)
# @cython.wraparound(False) 
# def matmul4(np.ndarray[dtype_t, ndim=2, mode="c"] A,
#             np.ndarray[dtype_t, ndim=2, mode="c"] B,
#             np.ndarray[dtype_t, ndim=2, mode="c"] C=None):
#     cdef int m = A.shape[0]
#     cdef int n = A.shape[1]
#     cdef int k = B.shape[1]
#     cdef dtype_t s
#     
#     my_dgemm(m, n, k, &A[0,0], &B[0,0], &C[0,0])
#                                                   
#     return C
# -

%timeit matmul4(A,B,C)


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise : Find prime numbers < 10000

# + {"slideshow": {"slide_type": "fragment"}}
# %load solutions/cython/is_prime0.py

def is_prime0(n):
    if n < 4: return True
    if n % 2 == 0 : return False
    k = 3
    while k*k <= n:
        if n % k == 0: return False
        k += 2
    return True


# + {"slideshow": {"slide_type": "fragment"}}
[ p for p in range(20) if is_prime0(p)]

# + {"slideshow": {"slide_type": "fragment"}}
L = list(range(10000))
%timeit [ p for p in L if is_prime0(p)]

# + {"slideshow": {"slide_type": "slide"}, "language": "cython"}
# def is_prime1(n):
#     if n < 4: return True
#     if n % 2 == 0 : return False
#     k = 3
#     while k*k <= n:
#         if n % k == 0: return False
#         k += 2
#     return True

# + {"slideshow": {"slide_type": "fragment"}}
[ p for p in range(20) if is_prime1(p)]

# + {"slideshow": {"slide_type": "fragment"}}
%timeit [p  for p in L if is_prime1(p)]

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Add Cython types without modifying the Python Code

# + {"slideshow": {"slide_type": "fragment"}, "language": "cython"}
# import cython
# @cython.locals(n=int, k=int)
# def is_prime2(n):
#     if n < 4: return True
#     if n % 2 == 0 : return False
#     k = 3
#     while k*k <= n:
#         if n % k == 0: return False
#         k += 2
#     return True

# + {"slideshow": {"slide_type": "fragment"}}
[ p for p in range(20) if is_prime2(p)]

# + {"slideshow": {"slide_type": "fragment"}}
%timeit [p for p in L if is_prime2(p) ]

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Cython function 

# + {"slideshow": {"slide_type": "fragment"}, "language": "cython"}
# import cython
# cdef bint is_prime3(int n):
#     if n < 4: return True
#     if n % 2 == 0: return False
#     cdef int k = 3
#     while k*k <= n:
#         if n % k == 0: return False
#         k += 2
#     return True
# def prime_list(L):
#     return [p for p in L if is_prime3(p)]

# + {"slideshow": {"slide_type": "fragment"}}
prime_list(list(range(20)))

# + {"slideshow": {"slide_type": "fragment"}}
%timeit prime_list(L)

# + {"slideshow": {"slide_type": "slide"}, "language": "cython"}
# import cython
# from numpy cimport ndarray
# import numpy
#
# cdef bint is_prime3(int n):
#     if n < 4: return True
#     if n % 2 == 0: return False
#     cdef int k = 3
#     while k*k <= n:
#         if n % k == 0: return False
#         k += 2
#     return True
#
# def prime_array(ndarray[int, ndim=1] L):
#     cdef ndarray[int, ndim=1] res = ndarray(shape=(L.shape[0]),dtype=numpy.int32)
#     cdef int i
#     for i in range(L.shape[0]):
#         res[i] = is_prime3(L[i])
#     return L[res==1]

# + {"slideshow": {"slide_type": "fragment"}}
import numpy as np
prime_array(np.arange(20,dtype=np.int32))

# + {"slideshow": {"slide_type": "fragment"}}
npL = numpy.array(L,dtype=np.int32)
%timeit prime_array(npL)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Using Parallelism
#
# * Cython supports native parallelism via OpenMP
# * by default, Python’s Global Interpreter Lock (GIL) prevents that several threads use the Python interpreter simultaneously
# * to use this kind of parallelism, the GIL must be released
#
# If you have a default compiler with openmp support you can use
# this magic command in your notebook.
# ```cython
# %%cython --compile-args=-fopenmp --link-args=-fopenmp
# ```

# + {"slideshow": {"slide_type": "slide"}}
%%file cython_omp.pyx
import cython
from cython.parallel cimport parallel, prange  # import parallel functions
import numpy as np
from numpy cimport ndarray

cdef bint is_prime4(int n) nogil:      #release the gil 
    if n < 4: return True
    if n % 2 == 0: return False
    cdef int k = 3
    while k*k <= n:
        if n % k == 0: return False
        k += 2
    return True

@cython.boundscheck(False)
def prime_array_omp(ndarray[int, ndim=1] L):
    cdef ndarray[int, ndim=1] res = ndarray(shape=(L.shape[0]),dtype=np.int32)
    cdef Py_ssize_t i
    with nogil, parallel(num_threads=4):
        for i in prange(L.shape[0]):     #Parallel loop
            res[i] = is_prime4(L[i])
    return L[res==1]


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# To use the OpenMP support, you need to enable OpenMP. For gcc this can be done as follows in a setup.py:

# + {"slideshow": {"slide_type": "fragment"}}
%%file setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os, sys
import numpy

if sys.platform == "darwin": # for omp, use gcc installed with brew
    os.environ["CC"]="gcc-10"
    os.environ["CXX"]="g++-10"

ext_modules = [
    Extension(
        "cython_omp",
        ["cython_omp.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='Cython OpenMP Example',
    ext_modules=cythonize(ext_modules),
)
# python setup.py build_ext --inplace

# + {"slideshow": {"slide_type": "slide"}}
%run setup.py build_ext --inplace --quiet

# + {"slideshow": {"slide_type": "fragment"}}
from cython_omp import prime_array_omp

# + {"slideshow": {"slide_type": "fragment"}}
prime_array_omp(np.arange(20,dtype=np.int32))

# + {"slideshow": {"slide_type": "fragment"}}
%timeit prime_array_omp(npL)

# + [markdown] {"slideshow": {"slide_type": "skip"}}
# ## References
# * [Cython documentation](http://docs.cython.org/en/latest/)
# * [An Interactive Introduction to Cython by Chris Swierczewski](http://www.cswiercz.info)
# * [Introduction To Python by Michael Kraus](http://michael-kraus.org/introduction-to-python.html)
# * [Cython by Xavier Juvigny 🇫🇷](http://calcul.math.cnrs.fr/IMG/pdf/cythontalk.pdf)
# * [Cython: C-Extensions for Python, Wiki](https://github.com/cython/cython/wiki)
# * Kurt W. Smith
#     - [Cython A Guide for Python Programmers](http://shop.oreilly.com/product/0636920033431.do)
#     - [Cython: Blend the Best of Python and C++ | SciPy 2015 Tutorial | Kurt Smith
# ](https://youtu.be/gMvkiQ-gOW8)
#     - [Cython: Speed up Python and NumPy, Pythonize C, C++, and Fortran, SciPy2013 Kurt W. Smith](https://youtu.be/JKCjsRDffXo)
#     - [SciPy 2017 - Cython by ](https://youtu.be/FepqwPI6U80)
#     - [Cython Book examples](https://github.com/cythonbook)
#     
# * [Parallel computing in Cython/threads - Neal Hughes](http://nealhughes.net/parallelcomp2/)
