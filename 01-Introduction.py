# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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

# # Introduction

# ![jkvdp_tweet](images/jkvdp_tweet.png)

# ## History
#
# - Project initiated by Guido Von Rossum in 1990
# - Interpreted language written in C.
# - Widely used in all domains (Web, Data Science, Scientific Computation).
# - This is a high level language with a simple syntax. 
# - Python types are numerously and powerful.
# - Bind Python with other languages is easy.
# - You can perform a lot of operations with very few lines.
# - Available on all platforms Unix, Windows, Mac OS X...
# - Many libraries offer Python bindings.
# - Python 2 is retired use only Python 3
#

# ## Python packages
#
# - [Awsome Python](https://awesome-python.com) : A curated list of awesome Python frameworks, libraries, software and resources.
# - [Python for Scientists](https://github.com/TomNicholas/Python-for-Scientists) : A list of recommended Python libraries, and resources, intended for scientific Python users.
# - [Weekly Python Newsletter](https://importpython.com/newsletter/)
# - [Real Python Tutorials](https://realpython.com)

# ## Performances
#
#  Python is not fast... but:
# - Sometimes it is. 
# - Most of operations are optimized.
# - Package like numpy can reduce the CPU time.
# - With Python you can save time to achieve your project.
#
# Some advices:
# - Write your program with Python language. 
# - If it is fast enough, be happy.
# - After profiling, optimize costly parts of your code.
#
# "Premature optimization is the root of all evil" (Donald Knuth 1974)
#

# ## Installation
#
# Conda is a powerful package manager and environment manager.
#
# [Ref: Getting started with conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
#
# - [Anaconda](https://www.anaconda.com/downloads) (large) 
# - [Miniconda](https://conda.io/miniconda.html) (small) 
# - [Miniforge](https://github.com/conda-forge/miniforge/releases) (best)

# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
# bash Miniforge3-Linux-x86_64.sh -b

# ##  Open a terminal (Linux/MacOSX) or a Anaconda prompt (Windows)

# Verify that conda is installed and running on your system by typing:

# ```bash
# ~/miniforge3/bin/conda init
# ```

# Conda displays the number of the version that you have installed.
#
# If you get an error message, make sure you closed and re-opened the
# terminal window after installing, or do it now. 
#
# To update conda to the current version. Type the following:
#
# ```bash
# conda update -y conda -n base
# ```

# ## Managing channels
#
# Conda channels are the locations where packages are stored. We use the [conda-forge](https://conda-forge.org),
# a good community-led collection of recipes for conda. If you installed [Miniforge](https://github.com/conda-forge/miniforge) you already have a conda specific to conda-forge.
#
# ```bash
# conda config --add channels conda-forge 
# conda config --set channel_priority strict
# ```
#
# Strict channel priority speed up conda operations and also reduce package incompatibility problems.

# ## Managing environments
#
# Conda allows you to create separate environments containing files, packages,
# and their dependencies that will not interact with other environments.
#
# When you begin using conda, you already have a default environment named
# ``base``. You don't want to put programs into your base environment, though.
# Create separate environments to keep your programs isolated from each other.
#
# ### Create a new environment and install a package in it.
#
# We will name the environment `test-env` and install the version 3.8 of `python`. At the Anaconda Prompt or in your terminal window, type the following:
# ```bash
# conda create -y -n test-env python=3.8
# ```

# ### To use, or "activate" the new environment, type the following:
#
# ```bash
# conda activate test-env
# ```

# Now that you are in your ``test-env`` environment, any conda commands you type will go to that environment until you deactivate it.
#
# Verify which version of Python is in your current environment:
#
# ```bash
# python --version
# ```

# ### To see a list of all your environments, type:

# conda info --envs

# The active environment is the one with an asterisk (*).
#
# ### Change your current environment back to the default (base):
#
# ```bash
# conda activate
# ```

# ## Managing packages
#
# - Check to see if a package you have not installed named "jupyter" is available from the Anaconda repository (must be connected to the Internet):

# conda search jupyter | grep conda-forge

# Conda displays a list of all packages with that name on conda-forge repository, so we know it is available.
#
# Install this package into the base environment:
#
# ```bash
# conda activate
# conda install -y jupyter -c conda-forge -n base
# ```

# Check to see if the newly installed program is in this environment:

# conda list jupyter

# ### Update a new conda environment from file
#
# Download the file [environment.yml](https://raw.githubusercontent.com/pnavaro/python-notebooks/master/environment.yml).
# This file contains the packages list for this course. Be aware that it takes time to download and install all packages.
#
# ```bash
# conda env update -f environment.yml -n test-env
# ```
#
# [Conda envs documentation](https://conda.io/docs/using/envs.html).

# Activating the conda environment will change your shell’s prompt to show what virtual environment you’re using, and modify the environment so that running python will get you that particular version and installation of Python. 
# <pre>
# $ conda activate test-env
# (test-env) $ python
# Python 3.6.2 (default, Jul 17 2017, 16:44:45) 
# [GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.42)] on darwin
# Type "help", "copyright", "credits" or "license" for more information.
# >>> quit()
# </pre>
#
# **You must do this everytime you open a new terminal**

# ## Install the kernel for jupyter
#
# ```bash
# conda run -n test-env python -m ipykernel install --user --name test-env
# ```
#
# With this command you create the `test-env` kernel with python and all course dependencies.
# The cell above will give you the path to the python that runs in this notebook.

import sys
print(f"{sys.executable}")

# jupyter-kernelspec list

# ## Mamba
#
# Mamba is a parallel reimplementation of the conda package manager in C++. It stays compatible as possible with conda interface. Install mamba from conda-forge:
# ```bash
# conda install mamba -c conda-forge
# ```
#
# To test it you can try to install the metapackage `r-tidyverse` which contains 144 packages.
#
# ```bash
# $ time conda create -y r-tidyverse -n condatest
# real	1m9.057s
# $ time mamba create -y r-tidyverse -n mambatest
# real	0m32.365s
# ```
# In this comparison packages are already downloaded, mamba is even better with downloads.

# + {"endofcell": "--"}
# ## Jupyter - Start The Notebook
#
# Open the notebook
# ```
# git clone https:://github.com/pnavaro/python-notebooks
# cd python-notebooks/notebooks
# jupyter notebook
# ```
# You should see the notebook open in your browser. If not, go to http://localhost:8888
#
# The Jupyter Notebook is an interactive environment for writing and running code. The notebook is capable of running code in a wide range of languages. However, each notebook is associated with Python3 kernel.
# -
# --

# ## Code cells allow you to enter and run code
#
# **Make a copy of this notebook by using the File menu.**
#
# Run a code cell using `Shift-Enter` or pressing the <button class='btn btn-default btn-xs'><i class="icon-step-forward fa fa-step-forward"></i></button> button in the toolbar above:
#
# There are two other keyboard shortcuts for running code:
#
# * `Alt-Enter` runs the current cell and inserts a new one below.
# * `Ctrl-Enter` run the current cell and enters command mode.

#
#
# ## Managing the Kernel
#
# Code is run in a separate process called the Kernel.  The Kernel can be interrupted or restarted.  Try running the following cell and then hit the <button class='btn btn-default btn-xs'><i class='icon-stop fa fa-stop'></i></button> button in the toolbar above.
#
# The "Cell" menu has a number of menu items for running code in different ways. These includes:
#
# * Run and Select Below
# * Run and Insert Below
# * Run All
# * Run All Above
# * Run All Below
#
#

# ## Restarting the kernels
#
# The kernel maintains the state of a notebook's computations. You can reset this state by restarting the kernel. This is done by clicking on the <button class='btn btn-default btn-xs'><i class='fa fa-repeat icon-repeat'></i></button> in the toolbar above.
#
#
# Check the [documentation](https://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Notebook%20Basics.html).

# ## First program
#
# - Print out the string "Hello world!" and its type.
# - Print out the value of `a` variable set to 6625 and its type.

s = "Hello World!"
print(type(s),s)

a = 6625
print(type(a),a)

# +
# a+s
# -

# ## Execute using python

# +
%%file hello.py

s = "Hello World!"
print(type(s),s)
a = 6625
print(type(a),a)
# -

# ```bash
# $ python3 hello.py
# <class 'str'> Hello World!
# <class 'int'> 6625
# ```

# ## Execute with ipython
# ```ipython
# (my-env) $ ipython
# Python 3.6.3 | packaged by conda-forge | (default, Nov  4 2017, 10:13:32)
# Type 'copyright', 'credits' or 'license' for more information
# IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
#
# In [1]: run hello.py
# <class 'str'> Hello World!
# <class 'int'> 6625
# ```

%run hello.py

# ## Python Types
# - Most of Python types are classes, typing is dynamic.
# - ; symbol can be used to split two Python commands on the same line.

s = int(2010); print(type(s))
s = 3.14; print(type(s))
s = True; print(type(s))
s = None; print(type(s))
s = 1.0j; print(type(s))
s = type(type(s)); print(type(s))

# ## Calculate with Python

x = 45        # This is a comment!
x += 2        # equivalent to x = x + 2
print(x, x > 45)

y = 2.5
print("x+y=",x+y, type(x+y))  # Add float to integer, result will be a float

print(x*10/y)   # true division returns a float
print(x*10//3)  # floor division discards the fractional part

print( x % 8) # the % operator returns the remainder of the division

print( f" x = {x:05d} ") # You can use C format rules to improve print output

# ## Multiple Assignment
# - Variables can simultaneously get new values. 
# - Expressions on the right-hand side are all evaluated first before assignments take place. 
# - The right-hand side expressions are evaluated from the left to the right.
# - Use it very carefully

a = b = c = 1
print(a, b, c) 

a, b, c = 1, 2, 3
print (a, b, c)

a, c = c, a     # Nice way to permute values
print (a, b, c) 

a < b < c, a > b > c

# ## `input` Function
#
# - Value returned by input is a string.
#
# - You must cast input call to get the type you want.
# ```py
# name = input("Please enter your name: ")
# x = int(input("Please enter an integer: "))
# L = list(input("Please enter 3 integers "))
# ```
# Copy-pase code above in three different cells and print returned values.
#
