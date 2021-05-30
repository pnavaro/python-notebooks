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
# # Semi-Lagrangian method
#
# Let us consider an abstract scalar advection equation of the form
#
# $$
# \frac{\partial f}{\partial t}+ a(x, t) \cdot \nabla f = 0. 
# $$
#
# The characteristic curves associated to this equation are the solutions of the ordinary differential equations
#
# $$
# \frac{dX}{dt} = a(X(t), t)
# $$
#
# We shall denote by $X(t, x, s)$ the unique solution of this equation associated to the initial condition $X(s) = x$.
#
# The classical semi-Lagrangian method is based on a backtracking of characteristics. Two steps are needed to update the distribution function $f^{n+1}$ at $t^{n+1}$ from its value $f^n$ at time $t^n$ :
# 1. For each grid point $x_i$ compute $X(t^n; x_i, t^{n+1})$ the value of the characteristic at $t^n$ which takes the value $x_i$ at $t^{n+1}$.
# 2. As the distribution solution of first equation verifies 
#
# $$f^{n+1}(x_i) = f^n(X(t^n; x_i, t^{n+1})),$$
#
# we obtain the desired value of $f^{n+1}(x_i)$ by computing $f^n(X(t^n;x_i,t^{n+1})$ by interpolation as $X(t^n; x_i, t^{n+1})$ is in general not a grid point.
#
# *[Eric Sonnendrücker - Numerical methods for the Vlasov equations](http://www-m16.ma.tum.de/foswiki/pub/M16/Allgemeines/NumMethVlasov/Num-Meth-Vlasov-Notes.pdf)*

# + {"slideshow": {"slide_type": "slide"}}
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 6.0)

# + {"slideshow": {"slide_type": "fragment"}}
# Disable the pager for lprun
from IPython.core import page
page.page = print


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Bspline interpolator
#
# - [De Boor's Algorithm - Wikipedia](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm)
#
# ### Numpy

# + {"slideshow": {"slide_type": "slide"}}
def bspline_python(p, j, x):
        """Return the value at x in [0,1[ of the B-spline with 
        integer nodes of degree p with support starting at j.
        Implemented recursively using the de Boor's recursion formula"""
        assert (x >= 0.0) & (x <= 1.0)
        assert (type(p) == int) & (type(j) == int)
        if p == 0:
            if j == 0:
                return 1.0
            else:
                return 0.0
        else:
            w = (x - j) / p
            w1 = (x - j - 1) / p
        return w * bspline_python(p - 1, j, x) + (1 - w1) * bspline_python(p - 1, j + 1, x)


# + {"slideshow": {"slide_type": "slide"}}
import numpy as np
from scipy.fftpack import fft, ifft 

class BSplineNumpy:
    
    """ Class to compute BSL advection of 1d function """
    
    def __init__(self, p, xmin, xmax, ncells):
        assert p & 1 == 1  # check that p is odd
        self.p = p
        self.ncells = ncells
        # compute eigenvalues of degree p b-spline matrix
        self.modes = 2 * np.pi * np.arange(ncells) / ncells
        self.deltax = (xmax - xmin) / ncells
        
        self.eig_bspl = bspline_python(p, -(p + 1) // 2, 0.0)
        for j in range(1, (p + 1) // 2):
            self.eig_bspl += bspline_python(p, j - (p + 1) // 2, 0.0) * 2 * np.cos(j * self.modes)
            
        self.eigalpha = np.zeros(ncells, dtype=complex)
    
    def interpolate_disp(self, f, alpha):
        """compute the interpolating spline of degree p of odd degree 
        of a function f on a periodic uniform mesh, at
        all points xi-alpha"""
        p = self.p
        assert (np.size(f) == self.ncells)
        # compute eigenvalues of cubic splines evaluated at displaced points
        ishift = np.floor(-alpha / self.deltax)
        beta = -ishift - alpha / self.deltax
        self.eigalpha.fill(0.)
        for j in range(-(p-1)//2, (p+1)//2 + 1):
            self.eigalpha += bspline_python(p, j-(p+1)//2, beta) * np.exp((ishift+j)*1j*self.modes)
            
        # compute interpolating spline using fft and properties of circulant matrices
        return np.real(ifft(fft(f) * self.eigalpha / self.eig_bspl))


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Interpolation test
# $\sin$ function after a displacement of alpha

# + {"slideshow": {"slide_type": "fragment"}}
def interpolation_test(BSplineClass):
    """ Test to check interpolation"""
    n = 64
    cs = BSplineClass(3,0,1,n)
    x = np.linspace(0,1,n, endpoint=False)
    f = np.sin(x*4*np.pi)
    alpha = 0.2
    return np.allclose(np.sin((x-alpha)*4*np.pi), cs.interpolate_disp(f, alpha))
    

interpolation_test(BSplineNumpy)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Profiling the code

# + {"slideshow": {"slide_type": "fragment"}}
%load_ext line_profiler

# + {"slideshow": {"slide_type": "fragment"}}
n =1024
cs = BSplineNumpy(3,0,1,n)
x = np.linspace(0,1,n, endpoint=False)
f = np.sin(x*4*np.pi)
alpha = 0.2;

# + {"slideshow": {"slide_type": "slide"}}
%lprun -s -f cs.interpolate_disp -T lp_results.txt cs.interpolate_disp(f, alpha);

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Fortran
#
# Replace the bspline computation by a fortran function, call it **bspline_fortran**.

# + {"slideshow": {"slide_type": "fragment"}}
%load_ext fortranmagic

# + {"slideshow": {"slide_type": "skip"}}
%%fortran
recursive function bspline_fortran(p, j, x) result(res)
    integer :: p, j
    real(8) :: x, w, w1
    real(8) :: res

    if (p == 0) then
        if (j == 0) then
            res = 1.0
            return
        else
            res = 0.0
            return
        end if
    else
        w = (x - j) / p
        w1 = (x - j - 1) / p
    end if
    
    res = w * bspline_fortran(p-1,j,x) &
    +(1-w1)*bspline_fortran(p-1,j+1,x)

end function bspline_fortran

# + {"slideshow": {"slide_type": "slide"}}
import numpy as np
from scipy.fftpack import fft, ifft

class BSplineFortran:
    
    def __init__(self, p, xmin, xmax, ncells):
        assert p & 1 == 1  # check that p is odd
        self.p = p
        self.ncells = ncells
        # compute eigenvalues of degree p b-spline matrix
        self.modes = 2 * np.pi * np.arange(ncells) / ncells
        self.deltax = (xmax - xmin) / ncells
        
        self.eig_bspl = bspline_fortran(p, -(p+1)//2, 0.0)
        for j in range(1, (p+1)//2):
            self.eig_bspl += bspline_fortran(p, j-(p+1)//2,0.0)*2*np.cos(j*self.modes)
            
        self.eigalpha = np.zeros(ncells, dtype=complex)
    
    def interpolate_disp(self, f, alpha):
        """compute the interpolating spline of degree p of odd degree 
        of a function f on a periodic uniform mesh, at
        all points xi-alpha"""
        p = self.p
        assert (np.size(f) == self.ncells)
        # compute eigenvalues of cubic splines evaluated at displaced points
        ishift = np.floor(-alpha / self.deltax)
        beta = -ishift - alpha / self.deltax
        self.eigalpha.fill(0.)
        for j in range(-(p-1)//2, (p+1)//2 + 1):
            self.eigalpha += bspline_fortran(p, j-(p+1)//2, beta) * np.exp((ishift+j)*1j*self.modes)
            
        # compute interpolating spline using fft and properties of circulant matrices
        return np.real(ifft(fft(f) * self.eigalpha / self.eig_bspl))



# + {"slideshow": {"slide_type": "slide"}}
interpolation_test(BSplineFortran)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Numba
#
# Create a optimized function of bspline python function with Numba. Call it bspline_numba.

# + {"slideshow": {"slide_type": "skip"}}
# %load solutions/landau_damping/bspline_numba.py
from numba import jit,  int32, float64
from scipy.fftpack import fft, ifft

@jit("float64(int32,int32,float64)",nopython=True)
def bspline_numba(p, j, x):
    
        """Return the value at x in [0,1[ of the B-spline with 
        integer nodes of degree p with support starting at j.
        Implemented recursively using the de Boor's recursion formula"""
        
        assert ((x >= 0.0) & (x <= 1.0))
        if p == 0:
            if j == 0:
                return 1.0
            else:
                return 0.0
        else:
            w = (x-j)/p
            w1 = (x-j-1)/p
        return w * bspline_numba(p-1,j,x)+(1-w1)*bspline_numba(p-1,j+1,x)


# -

class BSplineNumba:
    
    def __init__(self, p, xmin, xmax, ncells):
        assert p & 1 == 1  # check that p is odd
        self.p = p
        self.ncells = ncells
        # compute eigenvalues of degree p b-spline matrix
        self.modes = 2 * np.pi * np.arange(ncells) / ncells
        self.deltax = (xmax - xmin) / ncells
        
        self.eig_bspl = bspline_numba(p, -(p+1)//2, 0.0)
        for j in range(1, (p + 1) // 2):
            self.eig_bspl += bspline_numba(p,j-(p+1)//2,0.0)*2*np.cos(j*self.modes)
            
        self.eigalpha = np.zeros(ncells, dtype=complex)
        
    def interpolate_disp(self, f, alpha):
        """compute the interpolating spline of degree p of odd degree 
        of a function f on a periodic uniform mesh, at
        all points xi-alpha"""
        
        p = self.p
        assert (np.size(f) == self.ncells)
        # compute eigenvalues of cubic splines evaluated at displaced points
        ishift = np.floor(-alpha / self.deltax)
        beta = -ishift - alpha / self.deltax
        self.eigalpha.fill(0.)
        for j in range(-(p-1)//2, (p+1)//2+1):
            self.eigalpha += bspline_numba(p, j-(p+1)//2, beta)*np.exp((ishift+j)*1j*self.modes)
            
        # compute interpolating spline using fft and properties of circulant matrices
        return np.real(ifft(fft(f) * self.eigalpha / self.eig_bspl))


interpolation_test(BSplineNumba)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Pythran

# + {"slideshow": {"slide_type": "fragment"}}
import pythran

# + {"slideshow": {"slide_type": "fragment"}}
%load_ext pythran.magic


# + {"slideshow": {"slide_type": "fragment"}}
# %load solutions/landau_damping/bspline_pythran.py

#pythran export bspline_pythran(int,int,float64)
def bspline_pythran(p, j, x):
    if p == 0:
        if j == 0:
            return 1.0
        else:
            return 0.0
    else:
        w = (x-j)/p
        w1 = (x-j-1)/p
    return w * bspline_pythran(p-1,j,x)+(1-w1)*bspline_pythran(p-1,j+1,x)


# + {"slideshow": {"slide_type": "slide"}}
class BSplinePythran:
    
    def __init__(self, p, xmin, xmax, ncells):
        assert p & 1 == 1  # check that p is odd
        self.p = p
        self.ncells = ncells
        # compute eigenvalues of degree p b-spline matrix
        self.modes = 2 * np.pi * np.arange(ncells) / ncells
        self.deltax = (xmax - xmin) / ncells
        
        self.eig_bspl = bspline_pythran(p, -(p+1)//2, 0.0)
        for j in range(1, (p + 1) // 2):
            self.eig_bspl += bspline_pythran(p,j-(p+1)//2,0.0)*2*np.cos(j*self.modes)
            
        self.eigalpha = np.zeros(ncells, dtype=complex)
        
    def interpolate_disp(self, f, alpha):
        """compute the interpolating spline of degree p of odd degree 
        of a function f on a periodic uniform mesh, at
        all points xi-alpha"""
        
        p = self.p
        assert (f.size == self.ncells)
        # compute eigenvalues of cubic splines evaluated at displaced points
        ishift = np.floor(-alpha / self.deltax)
        beta = -ishift - alpha / self.deltax
        self.eigalpha.fill(0.)
        for j in range(-(p-1)//2, (p+1)//2+1):
            self.eigalpha += bspline_pythran(p, j-(p+1)//2, beta)*np.exp((ishift+j)*1j*self.modes)
            
        # compute interpolating spline using fft and properties of circulant matrices
        return np.real(ifft(fft(f) * self.eigalpha / self.eig_bspl))



# + {"slideshow": {"slide_type": "slide"}}
interpolation_test(BSplinePythran)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Cython
#
# - Create **bspline_cython** function.

# + {"slideshow": {"slide_type": "fragment"}}
%load_ext cython

# + {"slideshow": {"slide_type": "slide"}, "magic_args": "-a", "language": "cython"}
# def bspline_cython(p, j, x):
#         """Return the value at x in [0,1[ of the B-spline with 
#         integer nodes of degree p with support starting at j.
#         Implemented recursively using the de Boor's recursion formula"""
#         assert (x >= 0.0) & (x <= 1.0)
#         assert (type(p) == int) & (type(j) == int)
#         if p == 0:
#             if j == 0:
#                 return 1.0
#             else:
#                 return 0.0
#         else:
#             w = (x - j) / p
#             w1 = (x - j - 1) / p
#         return w * bspline_cython(p - 1, j, x) + (1 - w1) * bspline_cython(p - 1, j + 1, x)
#

# + {"slideshow": {"slide_type": "skip"}, "language": "cython"}
# import cython
# import numpy as np
# cimport numpy as np
# from scipy.fftpack import fft, ifft
#
# @cython.cdivision(True)
# cdef double bspline_cython(int p, int j, double x):
#         """Return the value at x in [0,1[ of the B-spline with 
#         integer nodes of degree p with support starting at j.
#         Implemented recursively using the de Boor's recursion formula"""
#         cdef double w, w1
#         if p == 0:
#             if j == 0:
#                 return 1.0
#             else:
#                 return 0.0
#         else:
#             w = (x - j) / p
#             w1 = (x - j - 1) / p
#         return w * bspline_cython(p-1,j,x)+(1-w1)*bspline_cython(p-1,j+1,x)
#
# class BSplineCython:
#     
#     def __init__(self, p, xmin, xmax, ncells):
#         self.p = p
#         self.ncells = ncells
#         # compute eigenvalues of degree p b-spline matrix
#         self.modes = 2 * np.pi * np.arange(ncells) / ncells
#         self.deltax = (xmax - xmin) / ncells
#         
#         self.eig_bspl = bspline_cython(p,-(p+1)//2, 0.0)
#         for j in range(1, (p + 1) // 2):
#             self.eig_bspl += bspline_cython(p,j-(p+1)//2,0.0)*2*np.cos(j*self.modes)
#             
#         self.eigalpha = np.zeros(ncells, dtype=complex)
#     
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     def interpolate_disp(self,  f,  alpha):
#         """compute the interpolating spline of degree p of odd degree 
#         of a function f on a periodic uniform mesh, at
#         all points xi-alpha"""
#         cdef Py_ssize_t j
#         cdef int p = self.p
#         # compute eigenvalues of cubic splines evaluated at displaced points
#         cdef int ishift = np.floor(-alpha / self.deltax)
#         cdef double beta = -ishift - alpha / self.deltax
#         self.eigalpha.fill(0)
#         for j in range(-(p-1)//2, (p+1)//2+1):
#             self.eigalpha += bspline_cython(p,j-(p+1)//2,beta)*np.exp((ishift+j)*1j*self.modes)
#             
#         # compute interpolating spline using fft and properties of circulant matrices
#         return np.real(ifft(fft(f) * self.eigalpha / self.eig_bspl))
#

# + {"slideshow": {"slide_type": "slide"}}
interpolation_test(BSplineCython)

# + {"slideshow": {"slide_type": "slide"}}
import seaborn; seaborn.set()
from tqdm.notebook import tqdm
Mrange = (2 ** np.arange(5, 10)).astype(int)

t_numpy = []
t_fortran = []
t_numba = []
t_pythran = []
t_cython = []

for M in tqdm(Mrange):
    x = np.linspace(0,1,M, endpoint=False)
    f = np.sin(x*4*np.pi)
    cs1 = BSplineNumpy(5,0,1,M)
    cs2 = BSplineFortran(5,0,1,M)
    cs3 = BSplineNumba(5,0,1,M)
    cs4 = BSplinePythran(5,0,1,M)
    cs5 = BSplineCython(5,0,1,M)
    
    alpha = 0.1
    t1 = %timeit -oq cs1.interpolate_disp(f, alpha)
    t2 = %timeit -oq cs2.interpolate_disp(f, alpha)
    t3 = %timeit -oq cs3.interpolate_disp(f, alpha)
    t4 = %timeit -oq cs4.interpolate_disp(f, alpha)
    t5 = %timeit -oq cs5.interpolate_disp(f, alpha)
    
    t_numpy.append(t1.best)
    t_fortran.append(t2.best)
    t_numba.append(t3.best)
    t_pythran.append(t4.best)
    t_cython.append(t5.best)

plt.loglog(Mrange, t_numpy, label='numpy')
plt.loglog(Mrange, t_fortran, label='fortran')
plt.loglog(Mrange, t_numba, label='numba')
plt.loglog(Mrange, t_pythran, label='pythran')
plt.loglog(Mrange, t_cython, label='cython')
plt.legend(loc='lower right')
plt.xlabel('Number of points')
plt.ylabel('Execution Time (s)');

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Vlasov-Poisson equation
# We consider the dimensionless Vlasov-Poisson equation for one species
# with a neutralizing background.
#
# $$ 
# \frac{\partial f}{\partial t}+ v\cdot \nabla_x f + E(t,x) \cdot \nabla_v f = 0, \\
# - \Delta \phi = 1 - \rho, E = - \nabla \phi \\
# \rho(t,x)  =  \int f(t,x,v)dv.
# $$
#
# - [Vlasov Equation - Wikipedia](https://en.wikipedia.org/wiki/Vlasov_equation)

# + {"slideshow": {"slide_type": "slide"}}
BSpline = dict(numpy=BSplineNumpy,
               fortran=BSplineFortran,
               cython=BSplineCython,
               numba=BSplineNumba,
               pythran=BSplinePythran)

class VlasovPoisson:
    
    def __init__(self, xmin, xmax, nx, vmin, vmax, nv, opt='numpy'):
        
        # Grid
        self.nx = nx
        self.x, self.dx = np.linspace(xmin, xmax, nx, endpoint=False, retstep=True)
        self.nv = nv
        self.v, self.dv = np.linspace(vmin, vmax, nv, endpoint=False, retstep=True)
        
        # Distribution function
        self.f = np.zeros((nx,nv)) 
        
        # Interpolators for advection
        BSplineClass = BSpline[opt]
        self.cs_x = BSplineClass(3, xmin, xmax, nx)
        self.cs_v = BSplineClass(3, vmin, vmax, nv)
        
        # Modes for Poisson equation
        self.modes = np.zeros(nx)
        k =  2* np.pi / (xmax - xmin)
        self.modes[:nx//2] = k * np.arange(nx//2)
        self.modes[nx//2:] = - k * np.arange(nx//2,0,-1)
        self.modes += self.modes == 0 # avoid division by zero 
        
    def advection_x(self, dt):
        for j in range(self.nv):
            alpha = dt * self.v[j]
            self.f[j,:] = self.cs_x.interpolate_disp(self.f[j,:], alpha)
            
    def advection_v(self, e, dt):
        for i in range(self.nx):
            alpha = dt * e[i] 
            self.f[:,i] = self.cs_v.interpolate_disp(self.f[:,i], alpha)
            
    def compute_rho(self):
        rho = self.dv * np.sum(self.f, axis=0)
        return  rho - rho.mean()
            
    def compute_e(self, rho):
        # compute Ex using that ik*Ex = rho
        rhok = fft(rho)/self.modes
        return np.real(ifft(-1j*rhok))
    
    def run(self, f, nstep, dt):
        self.f = f
        nrj = []
        self.advection_x(0.5*dt)
        for istep in tqdm(range(nstep)):
            rho = self.compute_rho()
            e = self.compute_e(rho)
            self.advection_v(e, dt)
            self.advection_x(dt)
            nrj.append( 0.5*np.log(np.sum(e*e)*self.dx))
                
        return nrj


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Landau Damping
#
# [Landau damping - Wikipedia](https://en.wikipedia.org/wiki/Landau_damping)

# + {"slideshow": {"slide_type": "slide"}}
from time import time

elapsed_time = {}
fig, axes = plt.subplots()
for opt in ('numpy', 'fortran', 'numba', 'cython','pythran'):
    
    # Set grid
    nx, nv = 32, 64
    xmin, xmax = 0.0, 4*np.pi
    vmin, vmax = -6., 6.
    
    # Create Vlasov-Poisson simulation
    sim = VlasovPoisson(xmin, xmax, nx, vmin, vmax, nv, opt=opt)

    # Initialize distribution function
    X, V = np.meshgrid(sim.x, sim.v)
    eps, kx = 0.001, 0.5
    f = (1.0+eps*np.cos(kx*X))/np.sqrt(2.0*np.pi)* np.exp(-0.5*V*V)

    # Set time domain
    nstep = 600
    t, dt = np.linspace(0.0, 60.0, nstep, retstep=True)
    
    # Run simulation
    etime = time()
    nrj = sim.run(f, nstep, dt)
    print(" {0:12s} : {1:.4f} ".format(opt, time()-etime))
    
    # Plot energy
    axes.plot(t, nrj, label=opt)

    
axes.plot(t, -0.1533*t-5.50)
plt.legend();

# + [markdown] {"slideshow": {"slide_type": "skip"}}
# ## References
# - [Optimizing Python with NumPy and Numba](https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/)
#
