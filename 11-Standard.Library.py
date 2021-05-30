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
# # Standard Library
#
# ## Operating System Interface
#

# + {"slideshow": {"slide_type": "fragment"}}
import os
os.getcwd()      # Return the current working directory

# + {"slideshow": {"slide_type": "fragment"}}
import sys
if sys.platform == "darwin":
    os.environ['CC']='gcc-10' # Change the default C compiler to gcc on macos
    
os.system('mkdir today') # Run the command mkdir in the system shell

# + {"slideshow": {"slide_type": "fragment"}}
os.chdir('today')   # Change current working directory
os.system('touch data.db') # Create the empty file data.db

# + {"slideshow": {"slide_type": "fragment"}}
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

# + {"slideshow": {"slide_type": "fragment"}}
import glob
glob.glob('*.py')


# + {"slideshow": {"slide_type": "fragment"}}
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

# + {"slideshow": {"slide_type": "fragment"}}
%run demo.py one two three

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Random

# + {"slideshow": {"slide_type": "fragment"}}
import random
random.choice(['apple', 'pear', 'banana'])

# + {"slideshow": {"slide_type": "fragment"}}
random.sample(range(100), 10)   # sampling without replacement

# + {"slideshow": {"slide_type": "fragment"}}
random.random()    # random float

# + {"slideshow": {"slide_type": "fragment"}}
random.randrange(6)    # random integer chosen from range(6)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Statistics

# + {"slideshow": {"slide_type": "fragment"}}
import statistics
data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
statistics.mean(data)

# + {"slideshow": {"slide_type": "fragment"}}
statistics.median(data)

# + {"slideshow": {"slide_type": "fragment"}}
statistics.variance(data)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Performance Measurement
#

# + {"slideshow": {"slide_type": "fragment"}}
from timeit import Timer
Timer('t=a; a=b; b=t', 'a=1; b=2').timeit()

# + {"slideshow": {"slide_type": "fragment"}}
Timer('a,b = b,a', 'a=1; b=2').timeit()

# + {"slideshow": {"slide_type": "fragment"}}
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

# + {"slideshow": {"slide_type": "fragment"}}
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
