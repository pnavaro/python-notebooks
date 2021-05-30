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
# # Errors and Exceptions
#
# There are two distinguishable kinds of errors: *syntax errors* and *exceptions*.
# - Syntax errors, also known as parsing errors, are the most common.
# - Exceptions are errors caused by statement or expression syntactically corrects.
# - Exceptions are not unconditionally fatal.
#
# [Exceptions in Python documentation](https://docs.python.org/3/library/exceptions.html)

# + {"slideshow": {"slide_type": "slide"}}
import sys
try:
    10 * (1/0)
except:
    print(sys.exc_info()[0])

# + {"slideshow": {"slide_type": "slide"}}
try:
    4 + spam*3
except:
    print(sys.exc_info()[0])

# + {"slideshow": {"slide_type": "slide"}}
try:
    '2' + 2
except:
    print(sys.exc_info()[0])

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Handling Exceptions
#
# - In example below, the user can interrupt the program with `Control-C` or the `stop` button in Jupyter Notebook.
# - Note that a user-generated interruption is signalled by raising the **KeyboardInterrupt** exception.
#

# + {"slideshow": {"slide_type": "fragment"}}
for s in ("0.1", "foo", "1000"):
   try:
     x = int(s)
     print(f' x = {x}')
     break
   except ValueError:
     print("Oops!  That was no valid number.  Try again...")

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - A try statement may have more than one except clause
# - The optional `else` clause must follow all except clauses.

# + {"slideshow": {"slide_type": "fragment"}}
import sys

def process_file(file):
    " Read the first line of f and convert to int and check if this integer is positive"
    try:
        i = int(open(file).readline().strip()) 
        print(i)
        assert i > 0
    except OSError as err:
        print(f"OS error: {err}")
    except ValueError:
        print("Could not convert data to an integer.")
    except:
        print("Unexpected error:", sys.exc_info()[0])

# Create the file workfile.txt
with open('workfile.txt','w') as f:
    f.write("foo")
    f.write("bar")

# + {"slideshow": {"slide_type": "slide"}}
process_file('workfile.txt') # catch exception return by int() call

# + {"slideshow": {"slide_type": "slide"}}
# Change permission of the file, workfile.txt cannot be read
!chmod u-r workfile.txt

# + {"slideshow": {"slide_type": "fragment"}}
process_file('workfile.txt') # catch exception return by open() call

# + {"slideshow": {"slide_type": "slide"}}
# Let's delete the file workfile.txt
!rm -f workfile.txt

# + {"slideshow": {"slide_type": "fragment"}}
process_file('workfile.txt') # catch another exception return by open() call

# + {"slideshow": {"slide_type": "slide"}}
# Insert the value -1 at the top of workfile.txt
!echo "-1" > workfile.txt
%cat workfile.txt

# + {"slideshow": {"slide_type": "slide"}}
process_file('workfile.txt') # catch exception return by assert()

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Raising Exceptions
#
# The raise statement allows the programmer to force a specified exception to occur.
#

# + {"slideshow": {"slide_type": "fragment"}}
try:
    raise NameError('HiThere')
except:
    print(sys.exc_info()[0])

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Defining Clean-up Actions
#
# - The try statement has an optional clause which is intended to define clean-up actions that must be executed under all circumstances.
#
# - A finally clause is always executed before leaving the try statement

# + {"slideshow": {"slide_type": "fragment"}}
try:
     raise KeyboardInterrupt
except:
    print(sys.exc_info()[0])
finally:
     print('Goodbye, world!')

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Wordcount Exercise
# - Improve the function `reduce` to read the results of `words` by using the `KeyError` exception to fill in the dictionary.
#  
