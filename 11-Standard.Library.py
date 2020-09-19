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
# # Standard Library
#
# ## Operating System Interface
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.842551Z", "iopub.execute_input": "2020-09-12T14:02:17.844712Z", "iopub.status.idle": "2020-09-12T14:02:17.848828Z", "shell.execute_reply": "2020-09-12T14:02:17.849382Z"}}
import os
os.getcwd()      # Return the current working directory

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.854967Z", "iopub.execute_input": "2020-09-12T14:02:17.855828Z", "iopub.status.idle": "2020-09-12T14:02:17.870318Z", "shell.execute_reply": "2020-09-12T14:02:17.870905Z"}}
import sys
if sys.platform == "darwin":
    os.environ['CC']='gcc-10' # Change the default C compiler to gcc on macos
    
os.system('mkdir today') # Run the command mkdir in the system shell

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.875000Z", "iopub.execute_input": "2020-09-12T14:02:17.875852Z", "iopub.status.idle": "2020-09-12T14:02:17.888747Z", "shell.execute_reply": "2020-09-12T14:02:17.889316Z"}}
os.chdir('today')   # Change current working directory
os.system('touch data.db') # Create the empty file data.db

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.894544Z", "iopub.execute_input": "2020-09-12T14:02:17.895458Z", "iopub.status.idle": "2020-09-12T14:02:17.897685Z", "shell.execute_reply": "2020-09-12T14:02:17.898263Z"}}
import shutil
shutil.copyfile('data.db', 'archive.db')
if os.path.exists('backup.db'):  # If file backup.db exists
    os.remove('backup.db')       # Remove it
shutil.move('archive.db', 'backup.db',)
shutil.os.chdir('..')

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## File Wildcards
#
# The glob module provides a function for making file lists from directory wildcard searches:

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.902089Z", "iopub.execute_input": "2020-09-12T14:02:17.902980Z", "iopub.status.idle": "2020-09-12T14:02:17.906801Z", "shell.execute_reply": "2020-09-12T14:02:17.907454Z"}}
import glob
glob.glob('*.py')


# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.914517Z", "iopub.execute_input": "2020-09-12T14:02:17.915689Z", "iopub.status.idle": "2020-09-12T14:02:17.916779Z", "shell.execute_reply": "2020-09-12T14:02:17.917343Z"}}
def recursive_replace( root, pattern, replace ) :
    """
    Function to replace a string inside a directory
    root : directory
    pattern : searched string
    replace "pattern" by "replace"
    """
    for directory, subdirs, filenames in os.walk( root ):
      for filename in filenames:
        path = os.path.join( directory, filename )
        text = open( path ).read()
        if pattern in text:
          print('occurence in :' + filename)
          open(path,'w').write( text.replace( pattern, replace ) )



# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Command Line Arguments
#
# These arguments are stored in the sys module’s argv attribute as a list.
# -

%%file demo.py
import sys
print(sys.argv)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.921694Z", "iopub.execute_input": "2020-09-12T14:02:17.922596Z", "iopub.status.idle": "2020-09-12T14:02:17.925132Z", "shell.execute_reply": "2020-09-12T14:02:17.925709Z"}}
%run demo.py one two three

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Random

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.930206Z", "iopub.execute_input": "2020-09-12T14:02:17.931060Z", "iopub.status.idle": "2020-09-12T14:02:17.933445Z", "shell.execute_reply": "2020-09-12T14:02:17.933981Z"}}
import random
random.choice(['apple', 'pear', 'banana'])

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.938947Z", "iopub.execute_input": "2020-09-12T14:02:17.939786Z", "shell.execute_reply": "2020-09-12T14:02:17.942082Z", "iopub.status.idle": "2020-09-12T14:02:17.942743Z"}}
random.sample(range(100), 10)   # sampling without replacement

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.946792Z", "iopub.execute_input": "2020-09-12T14:02:17.947753Z", "iopub.status.idle": "2020-09-12T14:02:17.950209Z", "shell.execute_reply": "2020-09-12T14:02:17.950776Z"}}
random.random()    # random float

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.954929Z", "iopub.execute_input": "2020-09-12T14:02:17.955873Z", "iopub.status.idle": "2020-09-12T14:02:17.958018Z", "shell.execute_reply": "2020-09-12T14:02:17.958675Z"}}
random.randrange(6)    # random integer chosen from range(6)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Statistics

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.962926Z", "iopub.execute_input": "2020-09-12T14:02:17.963785Z", "iopub.status.idle": "2020-09-12T14:02:17.972337Z", "shell.execute_reply": "2020-09-12T14:02:17.972920Z"}}
import statistics
data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
statistics.mean(data)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.977233Z", "iopub.execute_input": "2020-09-12T14:02:17.978111Z", "shell.execute_reply": "2020-09-12T14:02:17.980379Z", "iopub.status.idle": "2020-09-12T14:02:17.980993Z"}}
statistics.median(data)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:17.985304Z", "iopub.execute_input": "2020-09-12T14:02:17.986182Z", "iopub.status.idle": "2020-09-12T14:02:17.988582Z", "shell.execute_reply": "2020-09-12T14:02:17.989155Z"}}
statistics.variance(data)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Performance Measurement
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:18.032171Z", "iopub.execute_input": "2020-09-12T14:02:18.033085Z", "shell.execute_reply": "2020-09-12T14:02:18.035551Z", "iopub.status.idle": "2020-09-12T14:02:18.036082Z"}}
from timeit import Timer
Timer('t=a; a=b; b=t', 'a=1; b=2').timeit()

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:18.071011Z", "iopub.execute_input": "2020-09-12T14:02:18.072000Z", "iopub.status.idle": "2020-09-12T14:02:18.074375Z", "shell.execute_reply": "2020-09-12T14:02:18.074934Z"}}
Timer('a,b = b,a', 'a=1; b=2').timeit()

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:18.121691Z", "iopub.execute_input": "2020-09-12T14:02:18.164035Z", "iopub.status.idle": "2020-09-12T14:02:20.296945Z", "shell.execute_reply": "2020-09-12T14:02:20.297597Z"}}
%%timeit a=1; b=2
a,b = b,a


# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# The [profile](https://docs.python.org/3/library/profile.html#module-profile) and [pstats](https://docs.python.org/3/library/profile.html#module-pstats) modules provide tools for identifying time critical sections in larger blocks of code.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Quality Control
#
# One approach for developing high quality software is to write tests for each function.
#
# - The doctest module provides a tool for scanning a module and validating tests embedded in a program’s docstrings. 
# - This improves the documentation by providing the user with an example and it allows the doctest module to make sure the code remains true to the documentation:

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:02:20.302066Z", "iopub.execute_input": "2020-09-12T14:02:20.302923Z", "iopub.status.idle": "2020-09-12T14:02:21.696140Z", "shell.execute_reply": "2020-09-12T14:02:21.696741Z"}}
def average(values):
    """Computes the arithmetic mean of a list of numbers.

    >>> print(average([20, 30, 70]))
    40.0
    """
    return sum(values) / len(values)

import doctest
doctest.testmod()   # automatically validate the embedded tests

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Python’s standard library is very extensive
# - Containers and iterators: `collections`, `itertools`
# - Internet access: `urllib, email, mailbox, cgi, ftplib`
# - Dates and Times: `datetime, calendar, `
# - Data Compression: `zlib, gzip, bz2, lzma, zipfile, tarfile`
# - File formats: `csv, configparser, netrc, xdrlib, plistlib` 
# - Cryptographic Services: `hashlib, hmac, secrets`
# - Structure Markup Processing Tools: `html, xml`
#
# Check the [The Python Standard Library](https://docs.python.org/3/library/index.html)
