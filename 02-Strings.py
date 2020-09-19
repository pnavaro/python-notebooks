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
# # Strings

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.271712Z", "iopub.execute_input": "2020-09-12T13:29:11.272576Z", "iopub.status.idle": "2020-09-12T13:29:11.274193Z", "shell.execute_reply": "2020-09-12T13:29:11.274840Z"}}
word = "bonjour"

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.279370Z", "iopub.execute_input": "2020-09-12T13:29:11.280345Z", "iopub.status.idle": "2020-09-12T13:29:11.282951Z", "shell.execute_reply": "2020-09-12T13:29:11.283593Z"}}
print(word, len(word))

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# Add a `.` to the variable and then press `<TAB>` to get all attached methods available.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.295670Z", "iopub.execute_input": "2020-09-12T13:29:11.296678Z", "iopub.status.idle": "2020-09-12T13:29:11.299105Z", "shell.execute_reply": "2020-09-12T13:29:11.299815Z"}}
word.capitalize()

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# After choosing your method, press `shift+<TAB>` to get interface.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.304089Z", "iopub.execute_input": "2020-09-12T13:29:11.305144Z", "iopub.status.idle": "2020-09-12T13:29:11.307759Z", "shell.execute_reply": "2020-09-12T13:29:11.308354Z"}}
word.upper()

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.314144Z", "iopub.execute_input": "2020-09-12T13:29:11.315278Z", "iopub.status.idle": "2020-09-12T13:29:11.317792Z", "shell.execute_reply": "2020-09-12T13:29:11.318376Z"}}
help(word.replace) # or word.replace? 

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.322901Z", "iopub.execute_input": "2020-09-12T13:29:11.323852Z", "iopub.status.idle": "2020-09-12T13:29:11.326457Z", "shell.execute_reply": "2020-09-12T13:29:11.327049Z"}}
word.replace('o','O',1)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Strings and `print` Function
# Strings can be enclosed in single quotes ('...') or double quotes ("...") with the same result. \ can be used to escape quotes:
#
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.331690Z", "iopub.execute_input": "2020-09-12T13:29:11.332629Z", "iopub.status.idle": "2020-09-12T13:29:11.335221Z", "shell.execute_reply": "2020-09-12T13:29:11.335857Z"}}
print('spam eggs')          # single quotes
print('doesn\'t')           # use \' to escape the single quote...
print("doesn't")            # ...or use double quotes instead
print('"Yes," he said.')    #
print("\"Yes,\" he said.")
print('"Isn\'t," she said.')

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# `print` function translates C special characters

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.340046Z", "iopub.execute_input": "2020-09-12T13:29:11.341080Z", "iopub.status.idle": "2020-09-12T13:29:11.343540Z", "shell.execute_reply": "2020-09-12T13:29:11.344127Z"}}
s = '\tFirst line.\nSecond line.'  # \n means newline \t inserts tab
print(s)  # with print(), \n produces a new line
print(r'\tFirst line.\nSecond line.')  # note the r before the quote

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## String literals with multiple lines

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.348281Z", "iopub.execute_input": "2020-09-12T13:29:11.349191Z", "iopub.status.idle": "2020-09-12T13:29:11.351754Z", "shell.execute_reply": "2020-09-12T13:29:11.352358Z"}}
print("""\
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
""") 

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# \ character removes the initial newline.
#
# Strings can be concatenated (glued together) with the + operator, and repeated with *

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.356680Z", "iopub.execute_input": "2020-09-12T13:29:11.357683Z", "iopub.status.idle": "2020-09-12T13:29:11.360145Z", "shell.execute_reply": "2020-09-12T13:29:11.360783Z"}}
3 * ("Re" + 2 * 'n' + 'es ')

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# Two or more string literals next to each other are automatically concatenated.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.365019Z", "iopub.execute_input": "2020-09-12T13:29:11.366014Z", "iopub.status.idle": "2020-09-12T13:29:11.368506Z", "shell.execute_reply": "2020-09-12T13:29:11.369208Z"}}
text = ('Put several strings within parentheses '
         'to have them joined together.')
text

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# Strings can be indexed, with the first character having index 0. There is no separate character type; a character is simply a string of size one

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.373496Z", "iopub.execute_input": "2020-09-12T13:29:11.374347Z", "iopub.status.idle": "2020-09-12T13:29:11.377062Z", "shell.execute_reply": "2020-09-12T13:29:11.377673Z"}}
word = 'Python @ ENSAI'
print(word[0]) # character in position 0
print(word[5]) # character in position 5

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# Indices may also be negative numbers, to start counting from the right

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.381933Z", "iopub.execute_input": "2020-09-12T13:29:11.382921Z", "iopub.status.idle": "2020-09-12T13:29:11.385332Z", "shell.execute_reply": "2020-09-12T13:29:11.385922Z"}}
print(word[-1])  # last character
print(word[-2])  # second-last character

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Slicing Strings
# - Omitted first index defaults to zero, 
# - Omitted second index defaults to the size of the string being sliced.
# - Step can be set with the third index
#
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.390699Z", "iopub.execute_input": "2020-09-12T13:29:11.391702Z", "iopub.status.idle": "2020-09-12T13:29:11.393981Z", "shell.execute_reply": "2020-09-12T13:29:11.394608Z"}}
print(word[:2])  # character from the beginning to position 2 (excluded)
print(word[4:])  # characters from position 4 (included) to the end
print(word[-2:]) # characters from the second-last (included) to the end
print(word[::-1]) # This is the reversed string!

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.398749Z", "iopub.execute_input": "2020-09-12T13:29:11.399755Z", "iopub.status.idle": "2020-09-12T13:29:11.402313Z", "shell.execute_reply": "2020-09-12T13:29:11.402941Z"}}
word[::2]

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# Python strings cannot be changed — they are immutable.
# If you need a different string, you should create a new or use Lists.
#
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.407422Z", "iopub.execute_input": "2020-09-12T13:29:11.408429Z", "iopub.status.idle": "2020-09-12T13:29:11.410868Z", "shell.execute_reply": "2020-09-12T13:29:11.411449Z"}}
import sys
try:
    word[0] = 'J'
except:
    print(sys.exc_info()[0])

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.415276Z", "iopub.execute_input": "2020-09-12T13:29:11.416236Z", "iopub.status.idle": "2020-09-12T13:29:11.418402Z", "shell.execute_reply": "2020-09-12T13:29:11.419007Z"}}
## Some string methods
print(word.startswith('P'))

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.424525Z", "iopub.execute_input": "2020-09-12T13:29:11.425503Z", "iopub.status.idle": "2020-09-12T13:29:11.427794Z", "shell.execute_reply": "2020-09-12T13:29:11.428417Z"}}
print(*("\n"+w for w in dir(word) if not w.startswith('_')) )

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Exercise
#
# - Ask user to input a string.
# - Print out the string length.
# - Check if the last character is equal to the first character.
# - Check if this string contains only letters.
# - Check if this string is lower case.
# - Check if this string is a palindrome. A palindrome is a word, phrase, number, or other sequence of characters which reads the same backward as forward.

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:11.432160Z", "iopub.execute_input": "2020-09-12T13:29:11.433013Z", "iopub.status.idle": "2020-09-12T13:29:11.434919Z", "shell.execute_reply": "2020-09-12T13:29:11.435545Z"}}
# %load solutions/strings/demo.py

