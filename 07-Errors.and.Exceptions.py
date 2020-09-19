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
# # Errors and Exceptions
#
# There are two distinguishable kinds of errors: *syntax errors* and *exceptions*.
# - Syntax errors, also known as parsing errors, are the most common.
# - Exceptions are errors caused by statement or expression syntactically corrects.
# - Exceptions are not unconditionally fatal.
#
# [Exceptions in Python documentation](https://docs.python.org/3/library/exceptions.html)

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.252398Z", "iopub.execute_input": "2020-09-12T13:47:17.253376Z", "iopub.status.idle": "2020-09-12T13:47:17.256301Z", "shell.execute_reply": "2020-09-12T13:47:17.256911Z"}}
import sys
try:
    10 * (1/0)
except:
    print(sys.exc_info()[0])

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.261483Z", "iopub.execute_input": "2020-09-12T13:47:17.262395Z", "iopub.status.idle": "2020-09-12T13:47:17.264636Z", "shell.execute_reply": "2020-09-12T13:47:17.265243Z"}}
try:
    4 + spam*3
except:
    print(sys.exc_info()[0])

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.269419Z", "iopub.execute_input": "2020-09-12T13:47:17.270471Z", "iopub.status.idle": "2020-09-12T13:47:17.272891Z", "shell.execute_reply": "2020-09-12T13:47:17.273509Z"}}
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

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.278224Z", "iopub.execute_input": "2020-09-12T13:47:17.279110Z", "shell.execute_reply": "2020-09-12T13:47:17.281442Z", "iopub.status.idle": "2020-09-12T13:47:17.282055Z"}}
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

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.288636Z", "iopub.execute_input": "2020-09-12T13:47:17.289767Z", "iopub.status.idle": "2020-09-12T13:47:17.291519Z", "shell.execute_reply": "2020-09-12T13:47:17.292147Z"}}
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

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.296591Z", "iopub.execute_input": "2020-09-12T13:47:17.297608Z", "iopub.status.idle": "2020-09-12T13:47:17.299994Z", "shell.execute_reply": "2020-09-12T13:47:17.300586Z"}}
process_file('workfile.txt') # catch exception return by int() call

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.304772Z", "iopub.execute_input": "2020-09-12T13:47:17.305828Z", "iopub.status.idle": "2020-09-12T13:47:17.426916Z", "shell.execute_reply": "2020-09-12T13:47:17.427877Z"}}
# Change permission of the file, workfile.txt cannot be read
!chmod u-r workfile.txt

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.432284Z", "iopub.execute_input": "2020-09-12T13:47:17.433291Z", "iopub.status.idle": "2020-09-12T13:47:17.435847Z", "shell.execute_reply": "2020-09-12T13:47:17.436464Z"}}
process_file('workfile.txt') # catch exception return by open() call

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.440670Z", "iopub.execute_input": "2020-09-12T13:47:17.441594Z", "iopub.status.idle": "2020-09-12T13:47:17.560453Z", "shell.execute_reply": "2020-09-12T13:47:17.561111Z"}}
# Let's delete the file workfile.txt
!rm -f workfile.txt

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.565628Z", "iopub.execute_input": "2020-09-12T13:47:17.566570Z", "iopub.status.idle": "2020-09-12T13:47:17.568858Z", "shell.execute_reply": "2020-09-12T13:47:17.569496Z"}}
process_file('workfile.txt') # catch another exception return by open() call

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.574381Z", "iopub.execute_input": "2020-09-12T13:47:17.575264Z", "iopub.status.idle": "2020-09-12T13:47:17.811039Z", "shell.execute_reply": "2020-09-12T13:47:17.811762Z"}}
# Insert the value -1 at the top of workfile.txt
!echo "-1" > workfile.txt
%cat workfile.txt

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.816258Z", "iopub.execute_input": "2020-09-12T13:47:17.817261Z", "iopub.status.idle": "2020-09-12T13:47:17.819768Z", "shell.execute_reply": "2020-09-12T13:47:17.820525Z"}}
process_file('workfile.txt') # catch exception return by assert()

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Raising Exceptions
#
# The raise statement allows the programmer to force a specified exception to occur.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.825097Z", "iopub.execute_input": "2020-09-12T13:47:17.826112Z", "iopub.status.idle": "2020-09-12T13:47:17.828438Z", "shell.execute_reply": "2020-09-12T13:47:17.829044Z"}}
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

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:47:17.833605Z", "iopub.execute_input": "2020-09-12T13:47:17.834598Z", "iopub.status.idle": "2020-09-12T13:47:17.837110Z", "shell.execute_reply": "2020-09-12T13:47:17.837708Z"}}
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
