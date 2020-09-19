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
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# # What provide Numpy to Python ?
#
# - `ndarray` multi-dimensional array object
# - derived objects such as masked arrays and matrices
# - `ufunc` fast array mathematical operations.
# - Offers some Matlab-ish capabilities within Python
# - Initially developed by [Travis Oliphant](https://www.continuum.io/people/travis-oliphant).
# - Numpy 1.0 released October, 2006.
# - The [SciPy.org website](https://docs.scipy.org/doc/numpy) is very helpful.
# - NumPy fully supports an object-oriented approach.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Routines for fast operations on arrays.
#
# - shape manipulation
# - sorting
# - I/O
# - FFT
# - basic linear algebra
# - basic statistical operations
# - random simulation
# - statistics
# - and much more...

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Getting Started with NumPy
#
# - It is handy to import everything from NumPy into a Python console:
# ```python
# from numpy import *
# ```
# - But it is easier to read and debug if you use explicit imports.
# ```python
# import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt
# ```

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:11:49.976033Z", "iopub.execute_input": "2020-09-12T14:11:49.977870Z", "iopub.status.idle": "2020-09-12T14:11:50.104702Z", "shell.execute_reply": "2020-09-12T14:11:50.105535Z"}}
import numpy as np
print(np.__version__)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Why Arrays ?

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - Python lists are slow to process and use a lot of memory.
# - For tables, matrices, or volumetric data, you need lists of lists of lists... which becomes messy to program.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:11:50.110717Z", "iopub.execute_input": "2020-09-12T14:11:50.111971Z", "iopub.status.idle": "2020-09-12T14:11:50.113594Z", "shell.execute_reply": "2020-09-12T14:11:50.114436Z"}}
from random import random
from operator import truediv

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:11:50.161312Z", "iopub.execute_input": "2020-09-12T14:11:50.205672Z", "iopub.status.idle": "2020-09-12T14:11:55.713126Z", "shell.execute_reply": "2020-09-12T14:11:55.713779Z"}}
l1 = [random() for i in range(1000)]
l2 = [random() for i in range(1000)]
%timeit s = sum(map(truediv,l1,l2))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:11:55.719965Z", "iopub.execute_input": "2020-09-12T14:11:55.720947Z", "iopub.status.idle": "2020-09-12T14:12:07.618905Z", "shell.execute_reply": "2020-09-12T14:12:07.619718Z"}}
a1 = np.array(l1)
a2 = np.array(l2)
%timeit s = np.sum(a1/a2)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Numpy Arrays: The `ndarray` class.
#
# - There are important differences between NumPy arrays and Python lists:
#     - NumPy arrays have a fixed size at creation.
#     - NumPy arrays elements are all required to be of the same data type.
#     - NumPy arrays operations are performed in compiled code for performance.
# - Most of today's scientific/mathematical Python-based software use NumPy arrays.
# - NumPy gives us the code simplicity of Python, but the operation is speedily executed by pre-compiled C code.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.624586Z", "iopub.execute_input": "2020-09-12T14:12:07.625391Z", "iopub.status.idle": "2020-09-12T14:12:07.626927Z", "shell.execute_reply": "2020-09-12T14:12:07.627485Z"}}
a = np.array([0,1,2,3])  #  list
b = np.array((4,5,6,7))  #  tuple
c = np.matrix('8 9 0 1') #  string (matlab syntax)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.631969Z", "iopub.execute_input": "2020-09-12T14:12:07.632814Z", "iopub.status.idle": "2020-09-12T14:12:07.634867Z", "shell.execute_reply": "2020-09-12T14:12:07.635425Z"}}
print(a,b,c)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Element wise operations are the “default mode” 

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.646526Z", "iopub.execute_input": "2020-09-12T14:12:07.647429Z", "iopub.status.idle": "2020-09-12T14:12:07.650618Z", "shell.execute_reply": "2020-09-12T14:12:07.651148Z"}}
a*b,a+b

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.655677Z", "iopub.execute_input": "2020-09-12T14:12:07.656517Z", "iopub.status.idle": "2020-09-12T14:12:07.659134Z", "shell.execute_reply": "2020-09-12T14:12:07.659697Z"}}
5*a, 5+a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.664147Z", "iopub.execute_input": "2020-09-12T14:12:07.665075Z", "iopub.status.idle": "2020-09-12T14:12:07.667523Z", "shell.execute_reply": "2020-09-12T14:12:07.668106Z"}}
a @ b, np.dot(a,b)  # Matrix multiplication

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ##  NumPy Arrays Properties

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.672059Z", "iopub.execute_input": "2020-09-12T14:12:07.672843Z", "iopub.status.idle": "2020-09-12T14:12:07.674328Z", "shell.execute_reply": "2020-09-12T14:12:07.674887Z"}}
a = np.array([1,2,3,4,5]) # Simple array creation

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.679043Z", "iopub.execute_input": "2020-09-12T14:12:07.679972Z", "iopub.status.idle": "2020-09-12T14:12:07.682265Z", "shell.execute_reply": "2020-09-12T14:12:07.682892Z"}}
type(a) # Checking the type

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.686835Z", "iopub.execute_input": "2020-09-12T14:12:07.687760Z", "iopub.status.idle": "2020-09-12T14:12:07.690063Z", "shell.execute_reply": "2020-09-12T14:12:07.690615Z"}}
a.dtype # Print numeric type of elements

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.694503Z", "iopub.execute_input": "2020-09-12T14:12:07.695520Z", "iopub.status.idle": "2020-09-12T14:12:07.697899Z", "shell.execute_reply": "2020-09-12T14:12:07.698445Z"}}
a.itemsize # Print Bytes per element

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.702547Z", "iopub.execute_input": "2020-09-12T14:12:07.703354Z", "shell.execute_reply": "2020-09-12T14:12:07.705940Z", "iopub.status.idle": "2020-09-12T14:12:07.706481Z"}}
a.shape # returns a tuple listing the length along each dimension

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.710845Z", "iopub.execute_input": "2020-09-12T14:12:07.711703Z", "iopub.status.idle": "2020-09-12T14:12:07.713927Z", "shell.execute_reply": "2020-09-12T14:12:07.714504Z"}}
np.size(a), a.size # returns the entire number of elements.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.718628Z", "iopub.execute_input": "2020-09-12T14:12:07.719535Z", "shell.execute_reply": "2020-09-12T14:12:07.721881Z", "iopub.status.idle": "2020-09-12T14:12:07.722452Z"}}
a.ndim  # Number of dimensions

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.726239Z", "iopub.execute_input": "2020-09-12T14:12:07.727163Z", "shell.execute_reply": "2020-09-12T14:12:07.729638Z", "iopub.status.idle": "2020-09-12T14:12:07.730206Z"}}
a.nbytes # Memory used

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - ** Always use `shape` or `size` for numpy arrays instead of `len` **
# - `len` gives same information only for 1d array.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Functions to allocate arrays

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.735146Z", "iopub.execute_input": "2020-09-12T14:12:07.736058Z", "iopub.status.idle": "2020-09-12T14:12:07.738318Z", "shell.execute_reply": "2020-09-12T14:12:07.738874Z"}}
x = np.zeros((2,),dtype=('i4,f4,a10'))
x

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# `empty, empty_like, ones, ones_like, zeros, zeros_like, full, full_like`

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ##  Setting Array Elements Values

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.743270Z", "iopub.execute_input": "2020-09-12T14:12:07.744165Z", "iopub.status.idle": "2020-09-12T14:12:07.746283Z", "shell.execute_reply": "2020-09-12T14:12:07.746949Z"}}
a = np.array([1,2,3,4,5])
print(a.dtype)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.751334Z", "iopub.execute_input": "2020-09-12T14:12:07.752194Z", "iopub.status.idle": "2020-09-12T14:12:07.754678Z", "shell.execute_reply": "2020-09-12T14:12:07.755228Z"}}
a[0] = 10 # Change first item value
a, a.dtype

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.759469Z", "iopub.execute_input": "2020-09-12T14:12:07.760364Z", "iopub.status.idle": "2020-09-12T14:12:07.762820Z", "shell.execute_reply": "2020-09-12T14:12:07.763366Z"}}
a.fill(0) # slighty faster than a[:] = 0
a

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Setting Array Elements Types

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.768381Z", "iopub.execute_input": "2020-09-12T14:12:07.769398Z", "iopub.status.idle": "2020-09-12T14:12:07.771674Z", "shell.execute_reply": "2020-09-12T14:12:07.772416Z"}}
b = np.array([1,2,3,4,5.0]) # Last item is a float
b, b.dtype

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.777033Z", "iopub.execute_input": "2020-09-12T14:12:07.777962Z", "iopub.status.idle": "2020-09-12T14:12:07.780210Z", "shell.execute_reply": "2020-09-12T14:12:07.780755Z"}}
a.fill(3.0)  # assigning a float into a int array 
a[1] = 1.5   # truncates the decimal part
print(a.dtype, a)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.785410Z", "iopub.execute_input": "2020-09-12T14:12:07.786284Z", "iopub.status.idle": "2020-09-12T14:12:07.788535Z", "shell.execute_reply": "2020-09-12T14:12:07.789092Z"}}
a.astype('float64') # returns a new array containing doubles

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.793918Z", "iopub.execute_input": "2020-09-12T14:12:07.794796Z", "iopub.status.idle": "2020-09-12T14:12:07.797266Z", "shell.execute_reply": "2020-09-12T14:12:07.797800Z"}}
np.asfarray([1,2,3,4]) # Return an array converted to a float type

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Slicing x[lower:upper:step]
# - Extracts a portion of a sequence by specifying a lower and upper bound.
# - The lower-bound element is included, but the upper-bound element is **not** included.
# - The default step value is 1 and can be negative.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.801881Z", "iopub.execute_input": "2020-09-12T14:12:07.802798Z", "iopub.status.idle": "2020-09-12T14:12:07.804214Z", "shell.execute_reply": "2020-09-12T14:12:07.804867Z"}}
a = np.array([10,11,12,13,14])

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.810455Z", "iopub.execute_input": "2020-09-12T14:12:07.811362Z", "iopub.status.idle": "2020-09-12T14:12:07.813601Z", "shell.execute_reply": "2020-09-12T14:12:07.814160Z"}}
a[:2], a[-5:-3], a[0:2], a[-2:] # negative indices work

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.818869Z", "iopub.execute_input": "2020-09-12T14:12:07.819835Z", "iopub.status.idle": "2020-09-12T14:12:07.822275Z", "shell.execute_reply": "2020-09-12T14:12:07.822923Z"}}
a[::2], a[::-1]

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise: 
# - Compute derivative of $f(x) = \sin(x)$ with finite difference method.
# $$
#     \frac{\partial f}{\partial x} \sim \frac{f(x+dx)-f(x)}{dx}
# $$
#
# derivatives values are centered in-between sample points.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.827374Z", "iopub.execute_input": "2020-09-12T14:12:07.828288Z", "iopub.status.idle": "2020-09-12T14:12:07.829701Z", "shell.execute_reply": "2020-09-12T14:12:07.830282Z"}}
x, dx = np.linspace(0,4*np.pi,100, retstep=True)
y = np.sin(x)

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:07.836738Z", "iopub.execute_input": "2020-09-12T14:12:07.837567Z", "iopub.status.idle": "2020-09-12T14:12:08.703878Z", "shell.execute_reply": "2020-09-12T14:12:08.704434Z"}}
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12.,8.] # Increase plot size
plt.plot(x, np.cos(x),'b')
plt.title(r"$\rm{Derivative\ of}\ \sin(x)$");

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:08.765113Z", "iopub.execute_input": "2020-09-12T14:12:08.765926Z", "iopub.status.idle": "2020-09-12T14:12:08.911036Z", "shell.execute_reply": "2020-09-12T14:12:08.911593Z"}}
# Compute integral of x numerically
avg_height = 0.5*(y[1:]+y[:-1])
int_sin = np.cumsum(dx*avg_height)
plt.plot(x[1:], int_sin, 'ro', x, np.cos(0)-np.cos(x));

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Multidimensional array

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:08.916571Z", "iopub.execute_input": "2020-09-12T14:12:08.917511Z", "iopub.status.idle": "2020-09-12T14:12:08.918626Z", "shell.execute_reply": "2020-09-12T14:12:08.919411Z"}}
a = np.arange(4*3).reshape(4,3) # NumPy array
l = [[0,1,2],[3,4,5],[6,7,8],[9,10,11]] # Python List

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:08.923608Z", "iopub.execute_input": "2020-09-12T14:12:08.924402Z", "iopub.status.idle": "2020-09-12T14:12:08.926532Z", "shell.execute_reply": "2020-09-12T14:12:08.927093Z"}}
print(a)
print(l)

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:08.931424Z", "iopub.execute_input": "2020-09-12T14:12:08.932313Z", "shell.execute_reply": "2020-09-12T14:12:08.934678Z", "iopub.status.idle": "2020-09-12T14:12:08.935215Z"}}
l[-1][-1] # Access to last item

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:08.940015Z", "iopub.execute_input": "2020-09-12T14:12:08.940896Z", "iopub.status.idle": "2020-09-12T14:12:08.943067Z", "shell.execute_reply": "2020-09-12T14:12:08.943635Z"}}
print(a[-1,-1])  # Indexing syntax is different with NumPy array
print(a[0,0])    # returns the first item
print(a[1,:])    # returns the second line

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:08.948024Z", "iopub.execute_input": "2020-09-12T14:12:08.948920Z", "iopub.status.idle": "2020-09-12T14:12:08.951238Z", "shell.execute_reply": "2020-09-12T14:12:08.951787Z"}}
print(a[1]) # second line with 2d array
print(a[:,-1])  # last column

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise 
# - We compute numerically the Laplace Equation Solution using Finite Difference Method
# - Replace the computation of the discrete form of Laplace equation with numpy arrays
# $$
# T_{i,j} = \frac{1}{4} ( T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1})
# $$
# - The function numpy.allclose can help you to compute the residual.

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:08.959416Z", "iopub.execute_input": "2020-09-12T14:12:08.960232Z", "iopub.status.idle": "2020-09-12T14:12:59.256752Z", "shell.execute_reply": "2020-09-12T14:12:59.257302Z"}}
%%time
# Boundary conditions
Tnorth, Tsouth, Twest, Teast = 100, 20, 50, 50

# Set meshgrid
n, l = 64, 1.0
X, Y = np.meshgrid(np.linspace(0,l,n), np.linspace(0,l,n))
T = np.zeros((n,n))

# Set Boundary condition
T[n-1:, :] = Tnorth
T[:1, :]   = Tsouth
T[:, n-1:] = Teast
T[:, :1]   = Twest

residual = 1.0   
istep = 0
while residual > 1e-5 :
    istep += 1
    print ((istep, residual), end="\r")
    residual = 0.0   
    for i in range(1, n-1):
        for j in range(1, n-1):
            T_old = T[i,j]
            T[i, j] = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])
            if T[i,j]>0:
                residual=max(residual,abs((T_old-T[i,j])/T[i,j]))


print()
print("iterations = ",istep)
plt.title("Temperature")
plt.contourf(X, Y, T)
plt.colorbar()

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Arrays to ASCII files
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.261536Z", "iopub.execute_input": "2020-09-12T14:12:59.262323Z", "iopub.status.idle": "2020-09-12T14:12:59.263844Z", "shell.execute_reply": "2020-09-12T14:12:59.264407Z"}}
x = y = z = np.arange(0.0,5.0,1.0)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.269181Z", "iopub.execute_input": "2020-09-12T14:12:59.269997Z", "iopub.status.idle": "2020-09-12T14:12:59.394328Z", "shell.execute_reply": "2020-09-12T14:12:59.394982Z"}}
np.savetxt('test.out', (x,y,z), delimiter=',')   # X is an array
%cat test.out

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.400567Z", "iopub.execute_input": "2020-09-12T14:12:59.401392Z", "iopub.status.idle": "2020-09-12T14:12:59.522327Z", "shell.execute_reply": "2020-09-12T14:12:59.522932Z"}}
np.savetxt('test.out', (x,y,z), fmt='%1.4e')   # use exponential notation
%cat test.out

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Arrays from ASCII files

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.527873Z", "iopub.execute_input": "2020-09-12T14:12:59.528709Z", "iopub.status.idle": "2020-09-12T14:12:59.532731Z", "shell.execute_reply": "2020-09-12T14:12:59.533383Z"}}
np.loadtxt('test.out')

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - [save](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.save.html#numpy.save): Save an array to a binary file in NumPy .npy format
# - [savez](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.savez.html#numpy.savez) : Save several arrays into an uncompressed .npz archive
# - [savez_compressed](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed): Save several arrays into a compressed .npz archive
# - [load](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.load.html#numpy.load): Load arrays or pickled objects from .npy, .npz or pickled files.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## H5py
#
# Pythonic interface to the HDF5 binary data format. [h5py user manual](http://docs.h5py.org)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.538156Z", "iopub.execute_input": "2020-09-12T14:12:59.539042Z", "iopub.status.idle": "2020-09-12T14:12:59.607127Z", "shell.execute_reply": "2020-09-12T14:12:59.607720Z"}}
import h5py as h5

with h5.File('test.h5','w') as f:
    f['x'] = x
    f['y'] = y
    f['z'] = z

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.613542Z", "iopub.execute_input": "2020-09-12T14:12:59.614408Z", "iopub.status.idle": "2020-09-12T14:12:59.620531Z", "shell.execute_reply": "2020-09-12T14:12:59.621099Z"}}
with h5.File('test.h5','r') as f:
    for field in f.keys():
        print(field+':',np.array(f.get("x")))


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Slices Are References
# - Slices are references to memory in the original array.
# - Changing values in a slice also changes the original array.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.626086Z", "iopub.execute_input": "2020-09-12T14:12:59.627063Z", "iopub.status.idle": "2020-09-12T14:12:59.629593Z", "shell.execute_reply": "2020-09-12T14:12:59.630148Z"}}
a = np.arange(10)
b = a[3:6]
b  # `b` is a view of array `a` and `a` is called base of `b`

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.634490Z", "iopub.execute_input": "2020-09-12T14:12:59.635351Z", "iopub.status.idle": "2020-09-12T14:12:59.637674Z", "shell.execute_reply": "2020-09-12T14:12:59.638477Z"}}
b[0] = -1
a  # you change a view the base is changed.

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - Numpy does not copy if it is not necessary to save memory.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.643329Z", "iopub.execute_input": "2020-09-12T14:12:59.644224Z", "iopub.status.idle": "2020-09-12T14:12:59.646588Z", "shell.execute_reply": "2020-09-12T14:12:59.647129Z"}}
c = a[7:8].copy() # Explicit copy of the array slice
c[0] = -1 
a

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Fancy Indexing

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.652403Z", "iopub.execute_input": "2020-09-12T14:12:59.653473Z", "iopub.status.idle": "2020-09-12T14:12:59.655848Z", "shell.execute_reply": "2020-09-12T14:12:59.656407Z"}}
a = np.fromfunction(lambda i, j: (i+1)*10+j, (4, 5), dtype=int)
a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.660878Z", "iopub.execute_input": "2020-09-12T14:12:59.661763Z", "iopub.status.idle": "2020-09-12T14:12:59.664207Z", "shell.execute_reply": "2020-09-12T14:12:59.664779Z"}}
np.random.shuffle(a.flat) # shuffle modify only the first axis
a

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.669435Z", "iopub.execute_input": "2020-09-12T14:12:59.670361Z", "shell.execute_reply": "2020-09-12T14:12:59.672756Z", "iopub.status.idle": "2020-09-12T14:12:59.673330Z"}}
locations = a % 3 == 0 # locations can be used as a mask
a[locations] = 0 #set to 0 only the values that are divisible by 3
a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.677485Z", "iopub.execute_input": "2020-09-12T14:12:59.678308Z", "iopub.status.idle": "2020-09-12T14:12:59.680823Z", "shell.execute_reply": "2020-09-12T14:12:59.681474Z"}}
a += a == 0
a

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### `numpy.take`

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.685889Z", "iopub.execute_input": "2020-09-12T14:12:59.686780Z", "iopub.status.idle": "2020-09-12T14:12:59.689106Z", "shell.execute_reply": "2020-09-12T14:12:59.689757Z"}}
a[1:3,2:5]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.694530Z", "iopub.execute_input": "2020-09-12T14:12:59.695441Z", "iopub.status.idle": "2020-09-12T14:12:59.697815Z", "shell.execute_reply": "2020-09-12T14:12:59.698469Z"}}
np.take(a,[[6,7],[10,11]])  # Use flatten array indices

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Changing array shape

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.703144Z", "iopub.execute_input": "2020-09-12T14:12:59.704155Z", "iopub.status.idle": "2020-09-12T14:12:59.706607Z", "shell.execute_reply": "2020-09-12T14:12:59.707148Z"}}
grid = np.indices((2,3)) # Return an array representing the indices of a grid.
grid[0]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.711453Z", "iopub.execute_input": "2020-09-12T14:12:59.712307Z", "shell.execute_reply": "2020-09-12T14:12:59.714632Z", "iopub.status.idle": "2020-09-12T14:12:59.715161Z"}}
grid[1]

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.719501Z", "iopub.execute_input": "2020-09-12T14:12:59.720505Z", "iopub.status.idle": "2020-09-12T14:12:59.722985Z", "shell.execute_reply": "2020-09-12T14:12:59.723539Z"}}
grid.flat[:] # Return a view of grid array

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.728081Z", "iopub.execute_input": "2020-09-12T14:12:59.729010Z", "iopub.status.idle": "2020-09-12T14:12:59.731283Z", "shell.execute_reply": "2020-09-12T14:12:59.731843Z"}}
grid.flatten() # Return a copy

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.736644Z", "iopub.execute_input": "2020-09-12T14:12:59.737582Z", "iopub.status.idle": "2020-09-12T14:12:59.739951Z", "shell.execute_reply": "2020-09-12T14:12:59.740503Z"}}
np.ravel(grid, order='C') # A copy is made only if needed.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Sorting

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.745895Z", "iopub.execute_input": "2020-09-12T14:12:59.746774Z", "shell.execute_reply": "2020-09-12T14:12:59.749127Z", "iopub.status.idle": "2020-09-12T14:12:59.749738Z"}}
a=np.array([5,3,6,1,6,7,9,0,8])
np.sort(a) #. Return a view

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.754030Z", "iopub.execute_input": "2020-09-12T14:12:59.754892Z", "shell.execute_reply": "2020-09-12T14:12:59.757423Z", "iopub.status.idle": "2020-09-12T14:12:59.757997Z"}}
a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.762617Z", "iopub.execute_input": "2020-09-12T14:12:59.763580Z", "iopub.status.idle": "2020-09-12T14:12:59.765846Z", "shell.execute_reply": "2020-09-12T14:12:59.766515Z"}}
a.sort() # Change the array inplace
a

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Transpose-like operations

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.771148Z", "iopub.execute_input": "2020-09-12T14:12:59.771963Z", "iopub.status.idle": "2020-09-12T14:12:59.773580Z", "shell.execute_reply": "2020-09-12T14:12:59.774140Z"}}
a = np.array([5,3,6,1,6,7,9,0,8])
b = a
b.shape = (3,3) # b is a reference so a will be changed

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.778431Z", "iopub.execute_input": "2020-09-12T14:12:59.779340Z", "iopub.status.idle": "2020-09-12T14:12:59.781751Z", "shell.execute_reply": "2020-09-12T14:12:59.782370Z"}}
a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.786783Z", "iopub.execute_input": "2020-09-12T14:12:59.787797Z", "iopub.status.idle": "2020-09-12T14:12:59.790402Z", "shell.execute_reply": "2020-09-12T14:12:59.791031Z"}}
c = a.T # Return a view so a is not changed
np.may_share_memory(a,c)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.795614Z", "iopub.execute_input": "2020-09-12T14:12:59.796551Z", "iopub.status.idle": "2020-09-12T14:12:59.798832Z", "shell.execute_reply": "2020-09-12T14:12:59.799685Z"}}
c[0,0] = -1 # c is stored in same memory so change c you change a
a

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.804011Z", "iopub.execute_input": "2020-09-12T14:12:59.804989Z", "iopub.status.idle": "2020-09-12T14:12:59.807580Z", "shell.execute_reply": "2020-09-12T14:12:59.808140Z"}}
c  # is a transposed view of a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.812617Z", "iopub.execute_input": "2020-09-12T14:12:59.813522Z", "shell.execute_reply": "2020-09-12T14:12:59.815992Z", "iopub.status.idle": "2020-09-12T14:12:59.816565Z"}}
b  # b is a reference to a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.821149Z", "iopub.execute_input": "2020-09-12T14:12:59.822034Z", "iopub.status.idle": "2020-09-12T14:12:59.824425Z", "shell.execute_reply": "2020-09-12T14:12:59.824986Z"}}
c.base  # When the array is not a view `base` return None

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Methods Attached to NumPy Arrays

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.830099Z", "iopub.execute_input": "2020-09-12T14:12:59.831018Z", "iopub.status.idle": "2020-09-12T14:12:59.833468Z", "shell.execute_reply": "2020-09-12T14:12:59.834062Z"}}
a = np.arange(20).reshape(4,5)
np.random.shuffle(a.flat)
a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.839254Z", "iopub.execute_input": "2020-09-12T14:12:59.840129Z", "iopub.status.idle": "2020-09-12T14:12:59.842352Z", "shell.execute_reply": "2020-09-12T14:12:59.842883Z"}}
a -= a.mean()
a /= a.std() # Standardize the matrix

a.std(), a.mean()

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.847658Z", "iopub.execute_input": "2020-09-12T14:12:59.848531Z", "iopub.status.idle": "2020-09-12T14:12:59.850707Z", "shell.execute_reply": "2020-09-12T14:12:59.851238Z"}}
np.set_printoptions(precision=4)
print(a)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.855707Z", "iopub.execute_input": "2020-09-12T14:12:59.856735Z", "iopub.status.idle": "2020-09-12T14:12:59.859118Z", "shell.execute_reply": "2020-09-12T14:12:59.859673Z"}}
a.argmax() # max position in the memory contiguous array

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.864372Z", "iopub.execute_input": "2020-09-12T14:12:59.865282Z", "iopub.status.idle": "2020-09-12T14:12:59.867814Z", "shell.execute_reply": "2020-09-12T14:12:59.868515Z"}}
np.unravel_index(a.argmax(),a.shape) # get position in the matrix

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Array Operations over a given axis

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.872886Z", "iopub.execute_input": "2020-09-12T14:12:59.873676Z", "iopub.status.idle": "2020-09-12T14:12:59.875181Z", "shell.execute_reply": "2020-09-12T14:12:59.875743Z"}}
a = np.arange(20).reshape(5,4)
np.random.shuffle(a.flat)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.880387Z", "iopub.execute_input": "2020-09-12T14:12:59.881302Z", "iopub.status.idle": "2020-09-12T14:12:59.883613Z", "shell.execute_reply": "2020-09-12T14:12:59.884174Z"}}
a.sum(axis=0) # sum of each column

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.889178Z", "iopub.execute_input": "2020-09-12T14:12:59.890044Z", "iopub.status.idle": "2020-09-12T14:12:59.892455Z", "shell.execute_reply": "2020-09-12T14:12:59.893078Z"}}
np.apply_along_axis(sum, axis=0, arr=a)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.898053Z", "iopub.execute_input": "2020-09-12T14:12:59.898940Z", "iopub.status.idle": "2020-09-12T14:12:59.901380Z", "shell.execute_reply": "2020-09-12T14:12:59.901937Z"}}
np.apply_along_axis(sorted, axis=0, arr=a)

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# You can replace the `sorted` builtin fonction by a user defined function.

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.906381Z", "iopub.execute_input": "2020-09-12T14:12:59.907376Z", "iopub.status.idle": "2020-09-12T14:12:59.909795Z", "shell.execute_reply": "2020-09-12T14:12:59.910359Z"}}
np.empty(10)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.915505Z", "iopub.execute_input": "2020-09-12T14:12:59.916525Z", "iopub.status.idle": "2020-09-12T14:12:59.918741Z", "shell.execute_reply": "2020-09-12T14:12:59.919291Z"}}
np.linspace(0,2*np.pi,10)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.924004Z", "iopub.execute_input": "2020-09-12T14:12:59.925054Z", "iopub.status.idle": "2020-09-12T14:12:59.927473Z", "shell.execute_reply": "2020-09-12T14:12:59.928033Z"}}
np.arange(0,2.+0.4,0.4)

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.932951Z", "iopub.execute_input": "2020-09-12T14:12:59.933848Z", "iopub.status.idle": "2020-09-12T14:12:59.936076Z", "shell.execute_reply": "2020-09-12T14:12:59.936634Z"}}
np.eye(4)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.941200Z", "iopub.execute_input": "2020-09-12T14:12:59.942124Z", "iopub.status.idle": "2020-09-12T14:12:59.944583Z", "shell.execute_reply": "2020-09-12T14:12:59.945143Z"}}
a = np.diag(range(4))
a

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.949945Z", "iopub.execute_input": "2020-09-12T14:12:59.950856Z", "iopub.status.idle": "2020-09-12T14:12:59.953294Z", "shell.execute_reply": "2020-09-12T14:12:59.953916Z"}}
a[:,:,np.newaxis]

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Create the following arrays
# ```python
# [100 101 102 103 104 105 106 107 108 109]
# ```
# Hint: numpy.arange
# ```python
# [-2. -1.8 -1.6 -1.4 -1.2 -1. -0.8 -0.6 -0.4 -0.2 0. 
# 0.2 0.4 0.6 0.8 1. 1.2 1.4 1.6 1.8]
# ```
# Hint: numpy.linspace
# ```python
# [[ 0.001	0.00129155 0.0016681 0.00215443 0.00278256 
#      0.003593810.00464159 0.00599484 0.00774264 0.01]
# ```
# Hint: numpy.logspace
# ```python
# [[ 0. 0. -1. -1. -1.] 
#  [ 0. 0.  0. -1. -1.] 
#  [ 0. 0.  0.  0. -1.]
#  [ 0. 0.  0.  0.  0.]
#  [ 0. 0.  0.  0.  0.] 
#  [ 0. 0.  0.  0.  0.] 
#  [ 0. 0.  0.  0.  0.]]
# ```
# Hint: numpy.tri, numpy.zeros, numpy.transpose
#
# ```python
# [[ 0.  1.  2.  3. 4.] 
#  [-1.  0.  1.  2. 3.] 
#  [-1. -1.  0.  1. 2.] 
#  [-1. -1. -1.  0. 1.] 
#  [-1. -1. -1. -1. 0.]]
# ```
# Hint: numpy.ones, numpy.diag
#
# * Compute the integral numerically with Trapezoidal rule
# $$
# I = \int_{-\infty}^\infty e^{-v^2} dv
# $$
# with  $v \in [-10;10]$ and n=20.
#
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Views and Memory Management
# - If it exists one view of a NumPy array, it can be destroyed.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.958398Z", "iopub.execute_input": "2020-09-12T14:12:59.959238Z", "iopub.status.idle": "2020-09-12T14:12:59.966555Z", "shell.execute_reply": "2020-09-12T14:12:59.967112Z"}}
big = np.arange(1000000)
small = big[:5]
del big
small.base

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - Array called `big` is still allocated.
# - Sometimes it is better to create a copy.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.971457Z", "iopub.execute_input": "2020-09-12T14:12:59.972331Z", "iopub.status.idle": "2020-09-12T14:12:59.978937Z", "shell.execute_reply": "2020-09-12T14:12:59.979592Z"}}
big = np.arange(1000000)
small = big[:5].copy()
del big
print(small.base)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Change memory alignement

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.984181Z", "iopub.execute_input": "2020-09-12T14:12:59.985059Z", "iopub.status.idle": "2020-09-12T14:12:59.987281Z", "shell.execute_reply": "2020-09-12T14:12:59.987813Z"}}
del(a)
a = np.arange(20).reshape(5,4)
print(a.flags)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:12:59.992269Z", "iopub.execute_input": "2020-09-12T14:12:59.993137Z", "iopub.status.idle": "2020-09-12T14:12:59.995659Z", "shell.execute_reply": "2020-09-12T14:12:59.996385Z"}}
b = np.asfortranarray(a) # makes a copy
b.flags

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:00.001073Z", "iopub.execute_input": "2020-09-12T14:13:00.002201Z", "iopub.status.idle": "2020-09-12T14:13:00.004637Z", "shell.execute_reply": "2020-09-12T14:13:00.005226Z"}}
b.base is a

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# You can also create a fortran array with array function.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:00.010381Z", "iopub.execute_input": "2020-09-12T14:13:00.011368Z", "iopub.status.idle": "2020-09-12T14:13:00.012715Z", "shell.execute_reply": "2020-09-12T14:13:00.013384Z"}}
c = np.array([[1,2,3],[4,5,6]])
f = np.asfortranarray(c)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:00.018200Z", "iopub.execute_input": "2020-09-12T14:13:00.019046Z", "shell.execute_reply": "2020-09-12T14:13:00.021370Z", "iopub.status.idle": "2020-09-12T14:13:00.021957Z"}}
print(f.ravel(order='K')) # Return a 1D array using memory order
print(c.ravel(order='K')) # Copy is made only if necessary

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Broadcasting rules
#
# Broadcasting rules allow you to make an outer product between two vectors: the first method involves array tiling, the second one involves broadcasting. The last method is significantly faster.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:00.026781Z", "iopub.execute_input": "2020-09-12T14:13:00.027702Z", "iopub.status.idle": "2020-09-12T14:13:00.029132Z", "shell.execute_reply": "2020-09-12T14:13:00.029737Z"}}
n = 1000
a = np.arange(n)
ac = a[:, np.newaxis]   # column matrix
ar = a[np.newaxis, :]   # row matrix

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:00.040754Z", "iopub.execute_input": "2020-09-12T14:13:00.041590Z", "iopub.status.idle": "2020-09-12T14:13:13.194283Z", "shell.execute_reply": "2020-09-12T14:13:13.194869Z"}}
%timeit np.tile(a, (n,1)).T * np.tile(a, (n,1))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:13.199878Z", "iopub.execute_input": "2020-09-12T14:13:13.200789Z", "iopub.status.idle": "2020-09-12T14:13:15.203024Z", "shell.execute_reply": "2020-09-12T14:13:15.203611Z"}}
%timeit ac * ar

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.211134Z", "iopub.execute_input": "2020-09-12T14:13:15.211989Z", "iopub.status.idle": "2020-09-12T14:13:15.233987Z", "shell.execute_reply": "2020-09-12T14:13:15.234650Z"}}
np.all(np.tile(a, (n,1)).T * np.tile(a, (n,1)) == ac * ar)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Numpy Matrix
#
# Specialized 2-D array that retains its 2-D nature through operations. It has certain special operators, such as $*$ (matrix multiplication) and $**$ (matrix power).

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.239249Z", "iopub.execute_input": "2020-09-12T14:13:15.240157Z", "iopub.status.idle": "2020-09-12T14:13:15.242545Z", "shell.execute_reply": "2020-09-12T14:13:15.243105Z"}}
m = np.matrix('1 2; 3 4') #Matlab syntax
m

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.248032Z", "iopub.execute_input": "2020-09-12T14:13:15.248926Z", "iopub.status.idle": "2020-09-12T14:13:15.251257Z", "shell.execute_reply": "2020-09-12T14:13:15.251849Z"}}
a = np.matrix([[1, 2],[ 3, 4]]) #Python syntax
a

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.257015Z", "iopub.execute_input": "2020-09-12T14:13:15.257972Z", "iopub.status.idle": "2020-09-12T14:13:15.260337Z", "shell.execute_reply": "2020-09-12T14:13:15.261062Z"}}
a = np.arange(1,4)
b = np.mat(a) # 2D view, no copy!
b, np.may_share_memory(a,b)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.266027Z", "iopub.execute_input": "2020-09-12T14:13:15.266898Z", "iopub.status.idle": "2020-09-12T14:13:15.269457Z", "shell.execute_reply": "2020-09-12T14:13:15.270115Z"}}
a = np.matrix([[1, 2, 3],[ 3, 4, 5]])
a * b.T # Matrix vector product

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.274474Z", "iopub.execute_input": "2020-09-12T14:13:15.275403Z", "iopub.status.idle": "2020-09-12T14:13:15.277715Z", "shell.execute_reply": "2020-09-12T14:13:15.278414Z"}}
m * a # Matrix multiplication

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## StructuredArray using a compound data type specification

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.283346Z", "iopub.execute_input": "2020-09-12T14:13:15.284293Z", "iopub.status.idle": "2020-09-12T14:13:15.286512Z", "shell.execute_reply": "2020-09-12T14:13:15.287044Z"}}
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.292522Z", "iopub.execute_input": "2020-09-12T14:13:15.293334Z", "iopub.status.idle": "2020-09-12T14:13:15.295612Z", "shell.execute_reply": "2020-09-12T14:13:15.296144Z"}}
data['name'] = ['Pierre', 'Paul', 'Jacques', 'Francois']
data['age'] = [45, 10, 71, 39]
data['weight'] = [95.0, 75.0, 88.0, 71.0]
print(data)

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# ## RecordArray

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.300680Z", "iopub.execute_input": "2020-09-12T14:13:15.301602Z", "shell.execute_reply": "2020-09-12T14:13:15.303973Z", "iopub.status.idle": "2020-09-12T14:13:15.304501Z"}}
data_rec = data.view(np.recarray)
data_rec.age

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## NumPy Array Programming
# - Array operations are fast, Python loops are slow. 
# - Top priority: **avoid loops**
# - It’s better to do the work three times witharray operations than once with a loop.
# - This does require a change of habits.
# - This does require some experience.
# - NumPy’s array operations are designed to make this possible.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Fast Evaluation Of Array Expressions 
#
# - The `numexpr` package supplies routines for the fast evaluation of array expressions elementwise by using a vector-based virtual machine.
# - Expressions are cached, so reuse is fast.
#
# [Numexpr Users Guide](https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide)

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:13:15.336263Z", "iopub.execute_input": "2020-09-12T14:13:15.337098Z", "iopub.status.idle": "2020-09-12T14:13:57.631452Z", "shell.execute_reply": "2020-09-12T14:13:57.632083Z"}}
import numexpr as ne
import numpy as np
nrange = (2 ** np.arange(6, 24)).astype(int)

t_numpy = []
t_numexpr = []

for n in nrange:
    a = np.random.random(n)
    b = np.arange(n, dtype=np.double)
    c = np.random.random(n)
    
    c1 = ne.evaluate("a ** 2 + b ** 2 + 2 * a * b * c ", optimization='aggressive')

    t1 = %timeit -oq -n 10 a ** 2 + b ** 2 + 2 * a * b * c
    t2 = %timeit -oq -n 10 ne.re_evaluate()

    t_numpy.append(t1.best)
    t_numexpr.append(t2.best)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.loglog(nrange, t_numpy, label='numpy')
plt.loglog(nrange, t_numexpr, label='numexpr')

plt.legend(loc='lower right')
plt.xlabel('Vectors size')
plt.ylabel('Execution Time (s)');

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## References
# - [NumPy reference](http://docs.scipy.org/doc/numpy/reference/)
# - [Getting the Best Performance out of NumPy](http://ipython-books.github.io/featured-01/)
# - [Numpy by Konrad Hinsen](http://calcul.math.cnrs.fr/Documents/Ecoles/2013/python/NumPy%20avance.pdf)
