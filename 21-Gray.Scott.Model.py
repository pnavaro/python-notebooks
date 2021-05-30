# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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

# # Gray-Scott Model

import numpy as np

%config InlineBackend.figure_format = 'retina'


# The reaction-diffusion system described here involves two generic chemical species U and V, whose concentration at a given point in space is referred to by variables u and v. As the term implies, they react with each other, and they diffuse through the medium. Therefore the concentration of U and V at any given location changes with time and can differ from that at other locations.
#
# The overall behavior of the system is described by the following formula, two equations which describe three sources of increase and decrease for each of the two chemicals:
#
#
# $$
# \begin{array}{l}
# \displaystyle \frac{\partial u}{\partial t} = D_u \Delta u - uv^2 + F(1-u) \\
# \displaystyle \frac{\partial v}{\partial t} = D_v \Delta v + uv^2 - (F+k)v
# \end{array}
# $$
#
# The laplacian is computed with the following numerical scheme
#
# $$
# \Delta u_{i,j} \approx u_{i,j-1} + u_{i-1,j} -4u_{i,j} + u_{i+1, j} + u_{i, j+1}
# $$
#
# The classic Euler scheme is used to integrate the time derivative.

# ## Initialization
#
# $u$ is $1$ everywhere et $v$ is $0$ in the domain except in a square zone where $v = 0.25$ and $ u = 0.5$. This square located in the center of the domain is  $[0, 1]\times[0,1]$ with a size of $0.2$.
#

def init(n):
 
    u = np.ones((n+2,n+2))
    v = np.zeros((n+2,n+2))
    
    x, y = np.meshgrid(np.linspace(0, 1, n+2), np.linspace(0, 1, n+2))

    mask = (0.4<x) & (x<0.6) & (0.4<y) & (y<0.6)
    
    u[mask] = 0.50
    v[mask] = 0.25
        
    return u, v


# ## Boundary conditions
#
# We assume that the domain is periodic.
#

def periodic_bc(u):
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]


# ## Laplacian

def laplacian(u):
    """
    second order finite differences
    """
    return (                  u[ :-2, 1:-1] +
             u[1:-1, :-2] - 4*u[1:-1, 1:-1] + u[1:-1, 2:] +
                          +   u[2:  , 1:-1] )


# ## Gray-Scott model

def grayscott(U, V, Du, Dv, F, k):
    
    u, v = U[1:-1,1:-1], V[1:-1,1:-1]

    Lu = laplacian(U)
    Lv = laplacian(V)

    uvv = u*v*v
    u += Du*Lu - uvv + F*(1 - u)
    v += Dv*Lv + uvv - (F + k)*v

    periodic_bc(U)
    periodic_bc(V)


# ## Visualization
#
# Nous utiliserons les données suivantes.

Du, Dv = .1, .05
F, k = 0.0545, 0.062

# +
%%time
from tqdm.notebook import tqdm
from PIL import Image
U, V = init(300)

def create_image():
    global U, V
    for t in range(40):
        grayscott(U, V, Du, Dv, F, k)
    V_scaled = np.uint8(255*(V-V.min()) / (V.max()-V.min()))
    return V_scaled

def create_frames(n):

    return [create_image() for i in tqdm(range(n))]
    
frames = create_frames(500)

# +
from ipywidgets import interact, IntSlider

def display_sequence(iframe):
    
    return Image.fromarray(frames[iframe])
    
interact(display_sequence, 
         iframe=IntSlider(min=0,
                          max=len(frames)-1,
                          step=1,
                          value=0, 
                          continuous_update=True))
# -

import imageio
frames_scaled = [np.uint8(255 * frame) for frame in frames]
imageio.mimsave('images/movie.gif', frames_scaled, format='gif', fps=60)

from IPython.display import HTML
HTML('<img src="images/movie.gif">')


# ## References
#
# - [Reaction-Diffusion by the Gray-Scott Model: Pearson's Parametrization](https://mrob.com/pub/comp/xmorphia/)
