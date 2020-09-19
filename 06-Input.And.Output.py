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
# # Input and Output
# - str() function return human-readable representations of values.
# - repr() generate representations which can be read by the interpreter.
# - For objects which don’t have a particular representation for human consumption, str() will return the same value as repr().

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.711288Z", "iopub.execute_input": "2020-09-12T13:29:33.712341Z", "iopub.status.idle": "2020-09-12T13:29:33.715447Z", "shell.execute_reply": "2020-09-12T13:29:33.716039Z"}}
s = 'Hello, world.'
str(s)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.720626Z", "iopub.execute_input": "2020-09-12T13:29:33.721501Z", "shell.execute_reply": "2020-09-12T13:29:33.724086Z", "iopub.status.idle": "2020-09-12T13:29:33.724715Z"}}
l = list(range(4))
str(l)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.729179Z", "iopub.execute_input": "2020-09-12T13:29:33.730206Z", "iopub.status.idle": "2020-09-12T13:29:33.732773Z", "shell.execute_reply": "2020-09-12T13:29:33.733388Z"}}
repr(s)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.737445Z", "iopub.execute_input": "2020-09-12T13:29:33.738522Z", "iopub.status.idle": "2020-09-12T13:29:33.741053Z", "shell.execute_reply": "2020-09-12T13:29:33.741753Z"}}
repr(l)

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.746529Z", "iopub.execute_input": "2020-09-12T13:29:33.747496Z", "iopub.status.idle": "2020-09-12T13:29:33.749807Z", "shell.execute_reply": "2020-09-12T13:29:33.750398Z"}}
x = 10 * 3.25
y = 200 * 200
s = 'The value of x is ' + str(x) + ', and y is ' + repr(y) + '...'
print(s)

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# repr() of a string adds string quotes and backslashes:

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.754853Z", "iopub.execute_input": "2020-09-12T13:29:33.755806Z", "iopub.status.idle": "2020-09-12T13:29:33.758465Z", "shell.execute_reply": "2020-09-12T13:29:33.759078Z"}}
hello = 'hello, world\n'
hellos = repr(hello)
hellos

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# The argument to repr() may be any Python object:

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.763815Z", "iopub.execute_input": "2020-09-12T13:29:33.764778Z", "iopub.status.idle": "2020-09-12T13:29:33.767172Z", "shell.execute_reply": "2020-09-12T13:29:33.767802Z"}}
repr((x, y, ('spam', 'eggs')))

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.772700Z", "iopub.execute_input": "2020-09-12T13:29:33.773618Z", "iopub.status.idle": "2020-09-12T13:29:33.777891Z", "shell.execute_reply": "2020-09-12T13:29:33.778519Z"}}
n = 7
for x in range(1, n):
    for i in range(n):
        print(repr(x**i).rjust(i+2), end=' ') # rjust or center can be used
    print()

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.783427Z", "iopub.execute_input": "2020-09-12T13:29:33.785262Z", "iopub.status.idle": "2020-09-12T13:29:33.788404Z", "shell.execute_reply": "2020-09-12T13:29:33.789165Z"}}
for x in range(1, n):
    for i in range(n):
        print("%07d" % x**i, end=' ')  # old C format
    print()

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Usage of the `str.format()` method 

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.793464Z", "iopub.execute_input": "2020-09-12T13:29:33.794333Z", "iopub.status.idle": "2020-09-12T13:29:33.796723Z", "shell.execute_reply": "2020-09-12T13:29:33.797376Z"}}
print('We are at the {} in {}!'.format('ENSAI', 'Rennes'))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.801836Z", "iopub.execute_input": "2020-09-12T13:29:33.802691Z", "iopub.status.idle": "2020-09-12T13:29:33.804946Z", "shell.execute_reply": "2020-09-12T13:29:33.805571Z"}}
print('From {0} to  {1}'.format('September 7', 'September 14'))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.809857Z", "iopub.execute_input": "2020-09-12T13:29:33.810816Z", "iopub.status.idle": "2020-09-12T13:29:33.813250Z", "shell.execute_reply": "2020-09-12T13:29:33.813896Z"}}
print('It takes place at {place}'.format(place='Milon room'))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.818385Z", "iopub.execute_input": "2020-09-12T13:29:33.819369Z", "iopub.status.idle": "2020-09-12T13:29:33.821708Z", "shell.execute_reply": "2020-09-12T13:29:33.822315Z"}}
import math
print('The value of PI is approximately {:.7g}.'.format(math.pi))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Formatted string literals (Python 3.6)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.826255Z", "iopub.execute_input": "2020-09-12T13:29:33.827128Z", "iopub.status.idle": "2020-09-12T13:29:33.829614Z", "shell.execute_reply": "2020-09-12T13:29:33.830267Z"}}
print(f'The value of PI is approximately {math.pi:.4f}.')

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.834802Z", "iopub.execute_input": "2020-09-12T13:29:33.835729Z", "iopub.status.idle": "2020-09-12T13:29:33.838346Z", "shell.execute_reply": "2020-09-12T13:29:33.839006Z"}}
name = "Fred"
print(f"He said his name is {name}.")
print(f"He said his name is {name!r}.")

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.843599Z", "iopub.execute_input": "2020-09-12T13:29:33.844591Z", "iopub.status.idle": "2020-09-12T13:29:33.847116Z", "shell.execute_reply": "2020-09-12T13:29:33.847875Z"}}
f"He said his name is {repr(name)}."  # repr() is equivalent to !r

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.852815Z", "iopub.execute_input": "2020-09-12T13:29:33.853777Z", "iopub.status.idle": "2020-09-12T13:29:33.856279Z", "shell.execute_reply": "2020-09-12T13:29:33.856898Z"}}
width, precision = 10, 4
value = 12.34567
print(f"result: {value:{width}.{precision}f}")  # nested fields

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.861709Z", "iopub.execute_input": "2020-09-12T13:29:33.862577Z", "iopub.status.idle": "2020-09-12T13:29:33.864880Z", "shell.execute_reply": "2020-09-12T13:29:33.865493Z"}}
from datetime import *
today = datetime(year=2017, month=1, day=27)
print(f"{today:%B %d, %Y}")  # using date format specifier
# -

# <!-- #endregion -->

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Reading and Writing Files
#
# `open()` returns a file object, and is most commonly used with file name and accessing mode argument.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.870878Z", "iopub.execute_input": "2020-09-12T13:29:33.871792Z", "iopub.status.idle": "2020-09-12T13:29:33.993057Z", "shell.execute_reply": "2020-09-12T13:29:33.993694Z"}}
f = open('workfile.txt', 'w')
f.write("1. This is a txt file.\n")
f.write("2. \\n is used to begin a new line")
f.close()
!cat workfile.txt

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# `mode` can be :
# - 'r' when the file will only be read, 
# - 'w' for only writing (an existing file with the same name will be erased)
# - 'a' opens the file for appending; any data written to the file is automatically added to the end. 
# - 'r+' opens the file for both reading and writing. 
# - The mode argument is optional; 'r' will be assumed if it’s omitted.
# - Normally, files are opened in text mode.
# - 'b' appended to the mode opens the file in binary mode.

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:33.998481Z", "iopub.execute_input": "2020-09-12T13:29:33.999452Z", "iopub.status.idle": "2020-09-12T13:29:34.003307Z", "shell.execute_reply": "2020-09-12T13:29:34.004182Z"}}
with open('workfile.txt') as f:
    read_text = f.read()
f.closed

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.008493Z", "iopub.execute_input": "2020-09-12T13:29:34.009454Z", "iopub.status.idle": "2020-09-12T13:29:34.012321Z", "shell.execute_reply": "2020-09-12T13:29:34.012923Z"}}
read_text

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.017688Z", "iopub.execute_input": "2020-09-12T13:29:34.018772Z", "iopub.status.idle": "2020-09-12T13:29:34.023702Z", "shell.execute_reply": "2020-09-12T13:29:34.024444Z"}}
lines= []
with open('workfile.txt') as f:
    lines.append(f.readline())
    lines.append(f.readline())
    lines.append(f.readline())
    
lines

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - `f.readline()` returns an empty string when the end of the file has been reached.
# - `f.readlines()` or `list(f)` read all the lines of a file in a list.
#
# For reading lines from a file, you can loop over the file object. This is memory efficient, fast, and leads to simple code:

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.029078Z", "iopub.execute_input": "2020-09-12T13:29:34.030006Z", "iopub.status.idle": "2020-09-12T13:29:34.032351Z", "shell.execute_reply": "2020-09-12T13:29:34.032941Z"}}
with open('workfile.txt') as f:
    for line in f:
        print(line, end='')

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise: Wordcount Example
#
# [WordCount](https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html#Example:_WordCount_v1.0) is a simple application that counts the number of occurrences of each word in a given input set.
#
# - Use lorem module to write a text in the file "sample.txt"
# - Write a function `words` with file name as input that returns a sorted list of words present in the file.
# - Write the function `reduce` to read the results of words and sum the occurrences of each word to a final count, and then output the results as a dictionary
# `{word1:occurences1, word2:occurences2}`.
# - You can check the results using piped shell commands:
# ```sh
# cat sample.txt | fmt -1 | tr [:upper:] [:lower:] | tr -d '.' | sort | uniq -c 
# ```

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.037147Z", "iopub.execute_input": "2020-09-12T13:29:34.038041Z", "iopub.status.idle": "2020-09-12T13:29:34.043552Z", "shell.execute_reply": "2020-09-12T13:29:34.044432Z"}}
from lorem import text

text()


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.049100Z", "iopub.execute_input": "2020-09-12T13:29:34.050111Z", "iopub.status.idle": "2020-09-12T13:29:34.051663Z", "shell.execute_reply": "2020-09-12T13:29:34.052307Z"}}
def words( file ):
    """ Parse a file and returns a sorted list of words """
    pass

words('sample.txt')
#[('adipisci', 1),
# ('adipisci', 1),
# ('adipisci', 1),
# ('aliquam', 1),
# ('aliquam', 1),

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.056944Z", "iopub.execute_input": "2020-09-12T13:29:34.058020Z", "iopub.status.idle": "2020-09-12T13:29:34.060537Z", "shell.execute_reply": "2020-09-12T13:29:34.061175Z"}}
d = {}
d['word1'] = 3
d['word2'] = 2
d


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.065381Z", "iopub.execute_input": "2020-09-12T13:29:34.066230Z", "iopub.status.idle": "2020-09-12T13:29:34.068046Z", "shell.execute_reply": "2020-09-12T13:29:34.068670Z"}}
def reduce ( words ):
    """ Count the number of occurences of a word in list
    and return a dictionary """
    pass

reduce(words('sample.txt'))
#{'neque': 80),
# 'ut': 80,
# 'est': 76,
# 'amet': 74,
# 'magnam': 74,
# 'adipisci': 73,

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Saving structured data with json
#
# - JSON (JavaScript Object Notation) is a popular data interchange format.
# - JSON format is commonly used by modern applications to allow for data exchange. 
# - JSON can be used to communicate with applications written in other languages.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.073010Z", "iopub.execute_input": "2020-09-12T13:29:34.074098Z", "iopub.status.idle": "2020-09-12T13:29:34.076869Z", "shell.execute_reply": "2020-09-12T13:29:34.077478Z"}}
import json
json.dumps([1, 'simple', 'list'])

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.081862Z", "iopub.execute_input": "2020-09-12T13:29:34.082823Z", "iopub.status.idle": "2020-09-12T13:29:34.085296Z", "shell.execute_reply": "2020-09-12T13:29:34.085926Z"}}
x = dict(name="Pierre Navaro", organization="CNRS", position="IR")
with open('workfile.json','w') as f:
    json.dump(x, f)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.090025Z", "iopub.execute_input": "2020-09-12T13:29:34.091275Z", "iopub.status.idle": "2020-09-12T13:29:34.094764Z", "shell.execute_reply": "2020-09-12T13:29:34.095372Z"}}
with open('workfile.json','r') as f:
    x = json.load(f)
x

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:34.099719Z", "iopub.execute_input": "2020-09-12T13:29:34.100885Z", "iopub.status.idle": "2020-09-12T13:29:34.223217Z", "shell.execute_reply": "2020-09-12T13:29:34.223857Z"}}
%cat workfile.json

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# Use `ujson` for big data structures
# https://pypi.python.org/pypi/ujson
#
# For common file formats used in data science (CSV, xls, feather, parquet, ORC, HDF, avro,  ...) use packages like [pandas](https://pandas.pydata.org) or better [pyarrow](https://arrow.apache.org/docs/python/index.html). It depends of what you want to do with your data but [Dask](https://dask.org) and [pyspark](https://spark.apache.org/docs/latest/api/python/index.html) offer features to read and write (big) data files.
#
