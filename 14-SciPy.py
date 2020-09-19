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
# # Scipy
#
# ![scipy](images/scipyshiny_small.png)
#
# Scipy is the scientific Python ecosystem : 
# - fft, linear algebra, scientific computation,...
# - scipy contains numpy, it can be considered as an extension of numpy.
# - the add-on toolkits [Scikits](https://scikits.appspot.com/scikits) complements scipy.

# + {"slideshow": {"slide_type": "skip"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:15.936486Z", "iopub.execute_input": "2020-09-12T14:17:15.938394Z", "iopub.status.idle": "2020-09-12T14:17:16.429133Z", "shell.execute_reply": "2020-09-12T14:17:16.429710Z"}}
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ##  SciPy main packages
# - `constants` : Physical and mathematical constants
# - `fftpack` : Fast Fourier Transform routines
# - `integrate` : Integration and ordinary differential equation solvers
# - `interpolate` : Interpolation and smoothing splines
# - `io` : Input and Output
# - `linalg` : Linear algebra
# - `signal` : Signal processing
# - `sparse` : Sparse matrices and associated routines 
#

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:16.506164Z", "iopub.execute_input": "2020-09-12T14:17:16.507121Z", "iopub.status.idle": "2020-09-12T14:17:16.656919Z", "shell.execute_reply": "2020-09-12T14:17:16.657548Z"}}
from scipy.interpolate import interp1d 
x = np.linspace(-1, 1, num=5)  # 5 points evenly spaced in [-1,1].
y = (x-1.)*(x-0.5)*(x+0.5)     # x and y are numpy arrays
f0 = interp1d(x,y, kind='zero')
f1 = interp1d(x,y, kind='linear') 
f2 = interp1d(x,y, kind='quadratic') 
f3 = interp1d(x,y, kind='cubic') 
f4 = interp1d(x,y, kind='nearest') 

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:16.665449Z", "iopub.execute_input": "2020-09-12T14:17:16.698435Z", "iopub.status.idle": "2020-09-12T14:17:17.100571Z", "shell.execute_reply": "2020-09-12T14:17:17.101112Z"}}
xnew = np.linspace(-1, 1, num=40) 
ynew = (xnew-1.)*(xnew-0.5)*(xnew+0.5) 
plt.plot(x,y,'D',xnew,f0(xnew),':', xnew, f1(xnew),'-.',
                xnew,f2(xnew),'-',xnew ,f3(xnew),'--',
                xnew,f4(xnew),'--',xnew, ynew, linewidth=2)
plt.legend(['data','zero','linear','quadratic','cubic','nearest','exact'],
          loc='best');

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:17.243719Z", "iopub.execute_input": "2020-09-12T14:17:17.244691Z", "iopub.status.idle": "2020-09-12T14:17:17.246852Z", "shell.execute_reply": "2020-09-12T14:17:17.247393Z"}}
from scipy.interpolate import interp2d
x,y = sp.mgrid[0:1:20j,0:1:20j]  #create the grid 20x20
z = np.cos(4*sp.pi*x)*np.sin(4*sp.pi*y) #initialize the field
T1=interp2d(x,y,z,kind='linear') 
T2=interp2d(x,y,z,kind='cubic') 
T3=interp2d(x,y,z,kind='quintic')

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:17.255498Z", "iopub.execute_input": "2020-09-12T14:17:17.256326Z", "iopub.status.idle": "2020-09-12T14:17:17.974877Z", "shell.execute_reply": "2020-09-12T14:17:17.975517Z"}}
X,Y=sp.mgrid[0:1:100j,0:1:100j] #create the interpolation grid 100x100 
# complex -> number of points, float -> step size
plt.figure(1) 
plt.subplot(221) #Plot original data
plt.contourf(x,y,z) 
plt.title('20x20') 
plt.subplot(222) #Plot linear interpolation
plt.contourf(X,Y,T1(X[:,0],Y[0,:])) 
plt.title('100x100 linear') 
plt.subplot(223) #Plot cubic interpolation
plt.contourf(X,Y,T2(X[:,0],Y[0,:])) 
plt.title('100x100 cubic')
plt.subplot(224) #Plot quintic interpolation
plt.contourf(X,Y,T3(X[:,0],Y[0,:])) 
plt.title('100x100 quintic') 

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## FFT : scipy.fftpack
# - FFT dimension 1, 2 and n : fft, ifft (inverse), rfft (real), irfft, fft2 (dimension 2), ifft2, fftn (dimension n), ifftn.
# - Discrete cosinus transform : dct
# - Convolution product : convolve

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:17.982603Z", "iopub.execute_input": "2020-09-12T14:17:17.983607Z", "iopub.status.idle": "2020-09-12T14:17:22.623113Z", "shell.execute_reply": "2020-09-12T14:17:22.623685Z"}}
from numpy.fft import fft, ifft
x = np.random.random(2048)
%timeit ifft(fft(x))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:22.628695Z", "iopub.execute_input": "2020-09-12T14:17:22.629465Z", "iopub.status.idle": "2020-09-12T14:17:26.723442Z", "shell.execute_reply": "2020-09-12T14:17:26.724035Z"}}
from scipy.fftpack import fft, ifft
x = np.random.random(2048)
%timeit ifft(fft(x))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Linear algebra : scipy.linalg
# - Sovers, decompositions, eigen values. (same as numpy).
# - Matrix functions : expm, sinm, sinhm,...  
# - Block matrices diagonal, triangular, periodic,...

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.732815Z", "iopub.execute_input": "2020-09-12T14:17:26.733727Z", "iopub.status.idle": "2020-09-12T14:17:26.737386Z", "shell.execute_reply": "2020-09-12T14:17:26.737956Z"}}
import scipy.linalg as spl 
b=np.ones(5)
A=np.array([[1.,3.,0., 0.,0.],
           [ 2.,1.,-4, 0.,0.],
           [ 6.,1., 2,-3.,0.], 
           [ 0.,1., 4.,-2.,-3.], 
           [ 0.,0., 6.,-3., 2.]])
print("x=",spl.solve(A,b,sym_pos=False)) # LAPACK ( gesv ou posv )
AB=np.array([[0.,3.,-4.,-3.,-3.],
             [1.,1., 2.,-2., 2.],
             [2.,1., 4.,-3., 0.],
             [6.,1., 6., 0., 0.]])
print("x=",spl.solve_banded((2,1),AB,b)) # LAPACK ( gbsv )


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.744189Z", "iopub.execute_input": "2020-09-12T14:17:26.746272Z", "shell.execute_reply": "2020-09-12T14:17:26.749814Z", "iopub.status.idle": "2020-09-12T14:17:26.750732Z"}}
P,L,U = spl.lu(A) #  P A = L U
np.set_printoptions(precision=3)
for M in (P,L,U):
    print(M, end="\n"+20*"-"+"\n")

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ##  CSC (Compressed Sparse Column) 
#
# - All operations are optimized 
# - Efficient "slicing" along axis=1.
# - Fast Matrix-vector product.
# - Conversion to other format could be costly.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.809145Z", "iopub.execute_input": "2020-09-12T14:17:26.810468Z", "iopub.status.idle": "2020-09-12T14:17:26.814990Z", "shell.execute_reply": "2020-09-12T14:17:26.815744Z"}}
import scipy.sparse as spsp
row = np.array([0,2,2,0,1,2]) 
col = np.array([0,0,1,2,2,2])
data  = np.array([1,2,3,4,5,6]) 
Mcsc1 = spsp.csc_matrix((data,(row,col)),shape=(3,3)) 
Mcsc1.todense()

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.826295Z", "iopub.execute_input": "2020-09-12T14:17:26.827566Z", "iopub.status.idle": "2020-09-12T14:17:26.832313Z", "shell.execute_reply": "2020-09-12T14:17:26.833136Z"}}
indptr  = np.array([0,2,3,6]) 
indices = np.array([0,2,2,0,1,2]) 
data    = np.array([1,2,3,4,5,6]) 
Mcsc2 = spsp.csc_matrix ((data,indices,indptr),shape=(3,3))
Mcsc2.todense()

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Dedicated format for assembling 
# - `lil_matrix` : Row-based linked list matrix. Easy format to build your matrix and convert to other format before solving.
# - `dok_matrix` : A dictionary of keys based matrix. Ideal format for
# incremental matrix building. The conversion to csc/csr format is efficient.
# - `coo_matrix`  : coordinate list format. Fast conversion to formats CSC/CSR.
#
# [Lien vers la documentation scipy](http://docs.scipy.org/doc/scipy/reference/sparse.html)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Sparse matrices : [scipy.sparse.linalg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg)
#
# - speigen, speigen_symmetric, lobpcg : (ARPACK).
# - svd : (ARPACK).
# - Direct methods (UMFPACK or SUPERLU) or Krylov based methods 
# - Minimization : lsqr and minres
#
# For linear algebra:
# - Noobs: spsolve.
# - Intermmediate: dsolve.spsolve or isolve.spsolve
# - Advanced: splu, spilu (direct); cg, cgs, bicg, bicgstab, gmres, lgmres et qmr (iterative)
# - Boss: petsc4py et slepc4py.
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## LinearOperator
#
# The LinearOperator is used for matrix-free numerical methods.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.840147Z", "iopub.execute_input": "2020-09-12T14:17:26.841279Z", "iopub.status.idle": "2020-09-12T14:17:26.875505Z", "shell.execute_reply": "2020-09-12T14:17:26.876473Z"}}
import scipy.sparse.linalg as spspl
def mv(v):
   return np.array([2*v[0],3*v[1]])

A=spspl.LinearOperator((2 ,2),matvec=mv,dtype=float )
A

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.882818Z", "iopub.execute_input": "2020-09-12T14:17:26.884155Z", "iopub.status.idle": "2020-09-12T14:17:26.888584Z", "shell.execute_reply": "2020-09-12T14:17:26.889395Z"}}
A*np.ones(2)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.896480Z", "iopub.execute_input": "2020-09-12T14:17:26.897764Z", "iopub.status.idle": "2020-09-12T14:17:26.902265Z", "shell.execute_reply": "2020-09-12T14:17:26.903078Z"}}
A.matmat(np.array([[1,-2],[3,6]]))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## LU decomposition

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.912563Z", "iopub.execute_input": "2020-09-12T14:17:26.913782Z", "iopub.status.idle": "2020-09-12T14:17:26.920074Z", "shell.execute_reply": "2020-09-12T14:17:26.920886Z"}}
N = 50
un = np.ones(N)
w = np.random.rand(N+1)
A = spsp.spdiags([w[1:],-2*un,w[:-1]],[-1,0,1],N,N) # tridiagonal matrix
A = A.tocsc()
b = un
op = spspl.splu(A)
op

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.926070Z", "iopub.execute_input": "2020-09-12T14:17:26.927415Z", "iopub.status.idle": "2020-09-12T14:17:26.931809Z", "shell.execute_reply": "2020-09-12T14:17:26.932624Z"}}
x=op.solve(b)
spl.norm(A*x-b)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Conjugate Gradient

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.941001Z", "iopub.execute_input": "2020-09-12T14:17:26.946217Z", "iopub.status.idle": "2020-09-12T14:17:26.949574Z", "shell.execute_reply": "2020-09-12T14:17:26.950147Z"}}
global k
k=0
def f(xk): # function called at every iterations
     global k
     print ("iteration {0:2d} residu = {1:7.3g}".format(k,spl.norm(A*xk-b)))
     k += 1

x,info=spspl.cg(A,b,x0=np.zeros(N),tol=1.0e-12,maxiter=N,M=None,callback=f)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Preconditioned conjugate gradient

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.954769Z", "iopub.execute_input": "2020-09-12T14:17:26.955871Z", "iopub.status.idle": "2020-09-12T14:17:26.959314Z", "shell.execute_reply": "2020-09-12T14:17:26.959905Z"}}
pc=spspl.spilu(A,drop_tol=0.1)  # pc is an ILU decomposition
xp=pc.solve(b)
spl.norm(A*xp-b)


# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.966049Z", "iopub.execute_input": "2020-09-12T14:17:26.966944Z", "iopub.status.idle": "2020-09-12T14:17:26.973399Z", "shell.execute_reply": "2020-09-12T14:17:26.973984Z"}}
def mv(v):
    return pc.solve(v)
lo = spspl.LinearOperator((N,N),matvec=mv)
k = 0
x,info=spspl.cg(A,b,x0=np.zeros(N),tol=1.e-12,maxiter=N,M=lo,callback=f)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Numerical integration 
#
# - quad, dblquad, tplquad,... Fortran library QUADPACK.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:26.978341Z", "iopub.execute_input": "2020-09-12T14:17:26.979147Z", "iopub.status.idle": "2020-09-12T14:17:27.082255Z", "shell.execute_reply": "2020-09-12T14:17:27.082829Z"}}
import scipy.integrate as spi

x2=lambda x: x**2
4.**3/3  # int(x2) in [0,4]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:27.087471Z", "iopub.execute_input": "2020-09-12T14:17:27.088374Z", "iopub.status.idle": "2020-09-12T14:17:27.090667Z", "shell.execute_reply": "2020-09-12T14:17:27.091224Z"}}
spi.quad(x2,0.,4.)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Scipy ODE solver
#
# It uses the Fortran ODEPACK library. 
#
# ### Van der Pol Oscillator
# \begin{align}
# y_1'(t)	& = y_2(t), \nonumber \\
# y_2'(t)	& = 1000(1 - y_1^2(t))y_2(t) - y_1(t) \nonumber
# \end{align}
# $$
# with $y_1(0) = 2 $ and $ y_2(0) = 0. $.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:27.096802Z", "iopub.execute_input": "2020-09-12T14:17:27.097666Z", "iopub.status.idle": "2020-09-12T14:17:27.099006Z", "shell.execute_reply": "2020-09-12T14:17:27.099559Z"}}
import numpy as np
import scipy.integrate as spi

def vdp1000(y,t):
     dy=np.zeros(2)
     dy[0]=y[1]
     dy[1]=1000.*(1.-y[0]**2)*y[1]-y[0]
     return dy 


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:27.103955Z", "iopub.execute_input": "2020-09-12T14:17:27.104850Z", "iopub.status.idle": "2020-09-12T14:17:27.107325Z", "shell.execute_reply": "2020-09-12T14:17:27.107893Z"}}
t0, tf =0,  3000
N = 300000
t, dt = np.linspace(t0,tf,N, retstep=True)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:17:27.112488Z", "iopub.execute_input": "2020-09-12T14:17:27.113396Z", "iopub.status.idle": "2020-09-12T14:17:27.424693Z", "shell.execute_reply": "2020-09-12T14:17:27.425407Z"}}
y=spi.odeint(vdp1000,[2.,0.],t)
plt.plot(t,y[:,0]);
# -

# ## Exercise 
#
# The following code solve the Laplace equation using a dense matrix.
# - Modified the code to use a sparse matrix

# + {"execution": {"iopub.status.busy": "2020-09-12T14:17:27.462080Z", "iopub.execute_input": "2020-09-12T14:17:27.462973Z", "iopub.status.idle": "2020-09-12T14:17:28.327353Z", "shell.execute_reply": "2020-09-12T14:17:28.328125Z"}}
%matplotlib inline
%config InlineBackend.figure_format = "retina"
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)

# Boundary conditions
Tnorth, Tsouth, Twest, Teast = 100, 20, 50, 50

# Set meshgrid
n = 50
l = 1.0
h = l / (n-1)
X, Y = np.meshgrid(np.linspace(0,l,n), np.linspace(0,l,n))
T = np.zeros((n,n),dtype='d')

# Set Boundary condition
T[n-1:, :] = Tnorth / h**2
T[:1, :] = Tsouth / h**2
T[:, n-1:] = Teast / h**2
T[:, :1] = Twest / h**2

A = np.zeros((n*n,n*n),dtype='d')
nn = n*n
ii = 0
for j in range(n):
    for i in range(n):   
      if j > 0:
         jj = ii - n
         A[ii,jj] = -1
      if j < n-1: 
         jj = ii + n
         A[ii,jj] = -1
      if i > 0:
         jj = ii - 1
         A[ii,jj] = -1
      if i < n-1:
         jj = ii + 1
         A[ii,jj] = -1
      A[ii,ii] = 4
      ii = ii+1


# + {"execution": {"iopub.status.busy": "2020-09-12T14:17:27.462080Z", "iopub.execute_input": "2020-09-12T14:17:27.462973Z", "iopub.status.idle": "2020-09-12T14:17:28.327353Z", "shell.execute_reply": "2020-09-12T14:17:28.328125Z"}}
%%time
U = np.linalg.solve(A,np.ravel(h**2*T))

# + {"execution": {"iopub.status.busy": "2020-09-12T14:17:27.462080Z", "iopub.execute_input": "2020-09-12T14:17:27.462973Z", "iopub.status.idle": "2020-09-12T14:17:28.327353Z", "shell.execute_reply": "2020-09-12T14:17:28.328125Z"}}
T = U.reshape(n,n)
plt.contourf(X,Y,T)
plt.colorbar()

# + {"execution": {"iopub.status.busy": "2020-09-12T14:17:28.336444Z", "iopub.execute_input": "2020-09-12T14:17:28.337307Z", "iopub.status.idle": "2020-09-12T14:17:29.339935Z", "shell.execute_reply": "2020-09-12T14:17:29.340588Z"}}
import scipy.sparse as spsp
import scipy.sparse.linalg as spspl

# Boundary conditions
Tnorth, Tsouth, Twest, Teast = 100, 20, 50, 50

# Set meshgrid
n = 50
l = 1.0
h = l / (n-1)
X, Y = np.meshgrid(np.linspace(0,l,n), np.linspace(0,l,n))
T = np.zeros((n,n),dtype='d')

# Set Boundary condition
T[n-1:, :]    = Tnorth / h**2
T[  :1, :]    = Tsouth / h**2
T[   :, n-1:] = Teast  / h**2
T[   :, :1]   = Twest  / h**2

bdiag = -4 * np.eye(n)
bup   = np.diag([1] * (n - 1), 1)
blow  = np.diag([1] * (n - 1), -1)
block = bdiag + bup + blow
# Creat a list of n blocks
blist = [block] * n
S = spsp.block_diag(blist)
# Upper diagonal array offset by -n
upper = np.diag(np.ones(n * (n - 1)), n)
# Lower diagonal array offset by -n
lower = np.diag(np.ones(n * (n - 1)), -n)
S += upper + lower

# + {"execution": {"iopub.status.busy": "2020-09-12T14:17:28.336444Z", "iopub.execute_input": "2020-09-12T14:17:28.337307Z", "iopub.status.idle": "2020-09-12T14:17:29.339935Z", "shell.execute_reply": "2020-09-12T14:17:29.340588Z"}}
%%time
U = sp.linalg.solve(S,np.ravel(h**2*T))

# + {"execution": {"iopub.status.busy": "2020-09-12T14:17:28.336444Z", "iopub.execute_input": "2020-09-12T14:17:28.337307Z", "iopub.status.idle": "2020-09-12T14:17:29.339935Z", "shell.execute_reply": "2020-09-12T14:17:29.340588Z"}}
plt.contourf(X,Y,U.reshape(n,n))
plt.colorbar();
