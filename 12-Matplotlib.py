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
# # Matplotlib
#
# - Python 2D plotting library which produces figures in many formats and interactive environments.
# - Tries to make easy things easy and hard things possible. 
# - You can generate plots, histograms, power spectra, bar charts, errorcharts, scatterplots, etc., with just a few lines of code. 
# - Check the [Matplotlib gallery](https://matplotlib.org/gallery.html).
# - For simple plotting the pyplot module provides a MATLAB-like interface, particularly when combined with IPython. 
# - Matplotlib provides a set of functions familiar to MATLAB users.
#
# *In this notebook we use some numpy command that will be explain more precisely later.*

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Line Plots
#  - `np.linspace(0,1,10)` return 10 evenly spaced values over $[0,1]$.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:32.536296Z", "iopub.execute_input": "2020-09-12T14:02:32.538280Z", "iopub.status.idle": "2020-09-12T14:02:35.259111Z", "shell.execute_reply": "2020-09-12T14:02:35.259753Z"}}
%matplotlib inline
# inline can be replaced by notebook to get interactive plots
import numpy as np
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = "retina"

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:35.304606Z", "iopub.execute_input": "2020-09-12T14:02:35.305430Z", "iopub.status.idle": "2020-09-12T14:02:35.654122Z", "shell.execute_reply": "2020-09-12T14:02:35.654661Z"}}
plt.rcParams['figure.figsize'] = (10.0, 6.0) # set figures display bigger
x = np.linspace(- 5*np.pi,5*np.pi,100) 
plt.plot(x,np.sin(x)/x);

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:35.836298Z", "iopub.execute_input": "2020-09-12T14:02:35.837250Z", "iopub.status.idle": "2020-09-12T14:02:35.840434Z", "shell.execute_reply": "2020-09-12T14:02:35.840998Z"}}
plt.plot(x,np.sin(x)/x,x,np.sin(2*x)/x);

# + [markdown] {"slideshow": {"slide_type": "skip"}}
# If you have a recent Macbook with a Retina screen, you can display high-resolution plot outputs.
# Running the next cell will give you double resolution plot output for Retina screens. 
#
# *Note: the example below won’t render on non-retina screens*

# + {"slideshow": {"slide_type": "skip"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:35.853483Z", "iopub.execute_input": "2020-09-12T14:02:35.854410Z", "iopub.status.idle": "2020-09-12T14:02:35.855817Z", "shell.execute_reply": "2020-09-12T14:02:35.856379Z"}}
%config InlineBackend.figure_format = 'retina'

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:36.017894Z", "iopub.execute_input": "2020-09-12T14:02:36.084223Z", "iopub.status.idle": "2020-09-12T14:02:36.089005Z", "shell.execute_reply": "2020-09-12T14:02:36.089548Z"}}
# red, dot-dash, triangles and blue, dot-dash, bullet
plt.plot(x,np.sin(x)/x, 'r-^',x,np.sin(2*x)/x, 'b-o');

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Simple Scatter Plot

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:36.119099Z", "iopub.execute_input": "2020-09-12T14:02:36.178526Z", "iopub.status.idle": "2020-09-12T14:02:36.379593Z", "shell.execute_reply": "2020-09-12T14:02:36.380160Z"}}
x = np.linspace(-1,1,50)
y = np.sqrt(1-x**2)
plt.scatter(x,y);

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Colormapped Scatter Plot

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:36.435959Z", "iopub.execute_input": "2020-09-12T14:02:36.440138Z", "iopub.status.idle": "2020-09-12T14:02:36.738293Z", "shell.execute_reply": "2020-09-12T14:02:36.738842Z"}}
theta = np.linspace(0,6*np.pi,50) # 50 steps from 0 to 6 PI
size = 30*np.ones(50) # array with 50 values set to 30
z = np.random.rand(50) # array with 50 random values in [0,1]
x = theta*np.cos(theta)
y = theta*np.sin(theta)
plt.scatter(x,y,size,z)
plt.colorbar();

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Change Colormap

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:36.791969Z", "iopub.execute_input": "2020-09-12T14:02:36.796047Z", "iopub.status.idle": "2020-09-12T14:02:37.021532Z", "shell.execute_reply": "2020-09-12T14:02:37.022106Z"}}
fig = plt.figure() # create a figure
ax = fig.add_subplot(1, 1, 1) # add a single plot
ax.scatter(x,y,size,z,cmap='jet');
ax.set_aspect('equal', 'datalim')

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# [colormaps](http://matplotlib.org/api/pyplot_summary.html?highlight=colormaps#matplotlib.pyplot.colormaps) in matplotlib documentation.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Multiple Figures

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:37.058305Z", "iopub.execute_input": "2020-09-12T14:02:37.060037Z", "iopub.status.idle": "2020-09-12T14:02:37.436792Z", "shell.execute_reply": "2020-09-12T14:02:37.437436Z"}}
plt.figure()
plt.plot(x)
plt.figure()
plt.plot(z,'ro');

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Multiple Plots Using `subplot`

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:37.629536Z", "iopub.execute_input": "2020-09-12T14:02:37.630416Z", "iopub.status.idle": "2020-09-12T14:02:37.837065Z", "shell.execute_reply": "2020-09-12T14:02:37.837703Z"}}
plt.subplot(1,2,1) # 1 row 1, 2 columns, active plot number 1
plt.plot(x,'b-*')
plt.subplot(1,2,2) # 1 row 1, 2 columns, active plot number 2
plt.plot(z,'ro');

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Legends
#  - Legends labels with plot

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:37.861619Z", "iopub.execute_input": "2020-09-12T14:02:37.946933Z", "iopub.status.idle": "2020-09-12T14:02:38.163293Z", "shell.execute_reply": "2020-09-12T14:02:38.163865Z"}}
theta =np.linspace(0,4*np.pi,200)
plt.plot(np.sin(theta), label='sin')
plt.plot(np.cos(theta), label='cos')
plt.legend();

# + [markdown] {"slideshow": {"slide_type": "subslide"}}
# - Labelling with `legend`

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:38.273004Z", "iopub.execute_input": "2020-09-12T14:02:38.281005Z", "iopub.status.idle": "2020-09-12T14:02:38.818425Z", "shell.execute_reply": "2020-09-12T14:02:38.818996Z"}}
plt.plot(np.sin(theta))
plt.plot(np.cos(theta)**2)
plt.legend(['sin','$\cos^2$']);

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Titles and Axis Labels

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:38.880285Z", "iopub.execute_input": "2020-09-12T14:02:38.886240Z", "iopub.status.idle": "2020-09-12T14:02:39.095155Z", "shell.execute_reply": "2020-09-12T14:02:39.095693Z"}}
plt.plot(theta,np.sin(theta))
plt.xlabel('radians from 0 to $4\pi$')
plt.ylabel('amplitude');

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:39.135696Z", "iopub.execute_input": "2020-09-12T14:02:39.136506Z", "iopub.status.idle": "2020-09-12T14:02:39.916533Z", "shell.execute_reply": "2020-09-12T14:02:39.917070Z"}}
t = np.arange(0.01, 20.0, 0.01)

plt.subplot(121) 
plt.semilogy(t, np.exp(-t/5.0))
plt.title('semilogy')
plt.grid(True)

plt.subplot(122,fc='y') 
plt.semilogx(t, np.sin(2*np.pi*t))
plt.title('semilogx')
plt.grid(True)


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Plot Grid and Save to File

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:40.013088Z", "iopub.execute_input": "2020-09-12T14:02:40.182917Z", "iopub.status.idle": "2020-09-12T14:02:40.187991Z", "shell.execute_reply": "2020-09-12T14:02:40.188577Z"}}
theta = np.linspace(0,2*np.pi,100)
plt.plot(np.cos(theta),np.sin(theta))
plt.grid();


# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:40.194998Z", "iopub.execute_input": "2020-09-12T14:02:40.195881Z", "iopub.status.idle": "2020-09-12T14:02:40.349860Z", "shell.execute_reply": "2020-09-12T14:02:40.350653Z"}}
plt.savefig('circle.png');
%ls *.png

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Histogram 

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:40.355486Z", "iopub.execute_input": "2020-09-12T14:02:40.436059Z", "iopub.status.idle": "2020-09-12T14:02:40.645397Z", "shell.execute_reply": "2020-09-12T14:02:40.645959Z"}}
from numpy.random import randn
plt.hist(randn(1000));

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# Change the number of bins and supress display of returned array with ;
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:40.650193Z", "iopub.execute_input": "2020-09-12T14:02:40.665470Z", "iopub.status.idle": "2020-09-12T14:02:40.903216Z", "shell.execute_reply": "2020-09-12T14:02:40.903864Z"}}
plt.hist(randn(1000), 30);

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Contour Plot

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:40.909342Z", "iopub.execute_input": "2020-09-12T14:02:40.910187Z", "shell.execute_reply": "2020-09-12T14:02:41.764479Z", "iopub.status.idle": "2020-09-12T14:02:41.765069Z"}}
x = y = np.arange(-2.0*np.pi, 2.0*np.pi+0.01, 0.01)
X, Y = np.meshgrid(x, y)
Z = np.sin(X)*np.cos(Y)

plt.contourf(X, Y, Z,cmap=plt.cm.hot);

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Image Display

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:41.769470Z", "iopub.execute_input": "2020-09-12T14:02:41.770263Z", "iopub.status.idle": "2020-09-12T14:02:44.119395Z", "shell.execute_reply": "2020-09-12T14:02:44.119938Z"}}
img = plt.imread("https://hackage.haskell.org/package/JuicyPixels-extra-0.1.0/src/data-examples/lenna.png")
plt.imshow(img)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## figure and axis
#
# Best method to create a plot with many components

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:44.151023Z", "iopub.execute_input": "2020-09-12T14:02:44.151894Z", "iopub.status.idle": "2020-09-12T14:02:44.370852Z", "shell.execute_reply": "2020-09-12T14:02:44.371455Z"}}
fig = plt.figure()
axis = fig.add_subplot(111, aspect='equal',
                     xlim=(-2, 2), ylim=(-2, 2))

state = -0.5 + np.random.random((50, 4))
state[:, :2] *= 3.9
bounds = [-1, 1, -1, 1]

particles = axis.plot(state[:,0], state[:,1], 'bo', ms=6)
rect = plt.Rectangle(bounds[::2],
                     bounds[1] - bounds[0],
                     bounds[3] - bounds[2],
                     ec='r', lw=2, fc='none')
axis.grid()
axis.add_patch(rect)
axis.text(-0.5,1.1,"BOX")

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Exercises
#
# Recreate the image my_plots.png using the *delicate_arch.png* file in *images* directory.
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Alternatives
#
# - [bqplot](https://github.com/bloomberg/bqplot/blob/master/README.md) : Jupyter Notebooks, Interactive.
# - [seaborn](https://seaborn.pydata.org) : Statistics build on top of matplotlib.
# - [toyplot](http://toyplot.readthedocs.io/en/stable/) : Nice graphes.
# - [bokeh](http://bokeh.pydata.org/en/latest/) : Interactive and Server mode.
# - [pygal](http://pygal.org/en/stable/) : Charting
# - [Altair](https://github.com/altair-viz/altair) : Data science (js backend)
# - [plot.ly](https://plot.ly/) : Data science and interactive
# - [Mayavi](http://code.enthought.com/projects/mayavi/): 3D 
# - [YT](http://yt-project.org): Astrophysics (volume rendering, contours, particles). 
# - [VisIt](http://www.visitusers.org/index.php?title=VisIt-tutorial-Python-scripting): Powerful, easy to use but heavy.
# - [Paraview](http://www.itk.org/Wiki/ParaView/Python_Scripting): The most-used visualization application. Need high learning effort.
# - [PyVista](https://docs.pyvista.org): 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)
# - [Yellowbrick](https://www.scikit-yb.org/en/latest/) : Yellowbrick: Machine Learning Visualization
# - [scikit-plot](https://scikit-plot.readthedocs.io/en/stable/) : Plot sklearn metrics

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:44.394119Z", "iopub.execute_input": "2020-09-12T14:02:44.394916Z", "iopub.status.idle": "2020-09-12T14:02:44.404167Z", "shell.execute_reply": "2020-09-12T14:02:44.404736Z"}}
#example from Filipe Fernandes
#http://nbviewer.jupyter.org/gist/ocefpaf/9730c697819e91b99f1d694983e39a8f
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

g = 9.81
denw = 1025.0  # Seawater density [kg/m**3].
sig = 7.3e-2  # Surface tension [N/m].
a = 1.0  # Wave amplitude [m].

L, h = 100.0, 50.0  # Wave height and water column depth.
k = 2 * np.pi / L
omega = np.sqrt((g * k + (sig / denw) * (k**3)) * np.tanh(k * h))
T = 2 * np.pi / omega
c = np.sqrt((g / k + (sig / denw) * k) * np.tanh(k * h))

# We'll solve the wave velocities in the `x` and `z` directions.
x, z = np.meshgrid(np.arange(0, 160, 10), np.arange(0, -80, -10),)
u, w = np.zeros_like(x), np.zeros_like(z)


def compute_vel(phase):
    u = a * omega * (np.cosh(k * (z+h)) / np.sinh(k*h)) * np.cos(k * x - phase)
    w = a * omega * (np.sinh(k * (z+h)) / np.sinh(k*h)) * np.sin(k * x - phase)
    mask = -z > h
    u[mask] = 0.0
    w[mask] = 0.0
    return u, w


def basic_animation(frames=91, interval=30, dt=0.3):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(xlim=(0, 150), ylim=(-70, 10))

    # Animated.
    quiver = ax.quiver(x, z, u, w, units='inches', scale=2)
    ax.quiverkey(quiver, 120, -60, 1,
                 label=r'1 m s$^{-1}$',
                 coordinates='data')
    line, = ax.plot([], [], 'b')

    # Non-animated.
    ax.plot([0, 150], [0, 0], 'k:')
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Distance [m]')
    text = (r'$\lambda$ = %s m;  h = %s m;  kh = %2.3f;  h/L = %s' %
            (L, h, k * h, h/L))
    ax.text(10, -65, text)
    time_step = ax.text(10, -58, '')
    line.set_data([], [])

    def init():
        return line, quiver, time_step

    def animate(i):
        time = i * dt
        phase = omega * time
        eta = a * np.cos(x[0] * k - phase)
        u, w = compute_vel(phase)
        quiver.set_UVC(u, w)
        line.set_data(x[0], 5 * eta)
        time_step.set_text('Time = {:.2f} s'.format(time))
        return line, quiver, time_step

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval)


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:44.427659Z", "iopub.execute_input": "2020-09-12T14:02:44.428474Z", "iopub.status.idle": "2020-09-12T14:02:55.921948Z", "shell.execute_reply": "2020-09-12T14:02:55.922766Z"}}
from IPython.display import HTML

HTML(basic_animation(dt=0.3).to_jshtml())

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## References
# - Simple examples with increasing difficulty https://matplotlib.org/examples/index.html
# - Gallery https://matplotlib.org/gallery.html
# - A [matplotlib tutorial](http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb), part of the [Lectures on Scientific Computing with Python](http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/tree/master) by [J.R. Johansson](https://github.com/jrjohansson).
# - [NumPy Beginner | SciPy 2016 Tutorial | Alexandre Chabot LeClerc](https://youtu.be/gtejJ3RCddE)
# - [matplotlib tutorial](http://www.loria.fr/~rougier/teaching/matplotlib) by Nicolas Rougier from LORIA.
# - [10 Useful Python Data Visualization Libraries for Any Discipline](https://blog.modeanalytics.com/python-data-visualization-libraries/)
