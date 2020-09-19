# ---
# jupyter:
#   jupytext:
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

# # Animation with matplotlib

%matplotlib inline


# +
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

# +
fig, ax = plt.subplots()

ax.set_xlim(( 0, 2))
ax.set_ylim((-2, 2))

line, = ax.plot([], [], lw=2)


# -

def init():
    line.set_data([], [])
    return (line,)


def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return (line,)


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=20, 
                               blit=True)

HTML(anim.to_html5_video())

HTML(anim.to_jshtml())

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
xdata, ydata = [], []
line1, = plt.plot([], [], 'r-', animated=True)


# +
def init():
    ax1.set_xlim((0,1))
    ax1.set_ylim((-1,1))
    return line1,

def update(frame):
    
    xdata.append(frame)
    ydata.append(np.sin(8*np.pi*frame))
    line1.set_data(xdata, ydata)
    
    return line1,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 1.0, 100), 
                    init_func=init, blit=True)

plt.rc('animation', html='html5')
ani
# -


