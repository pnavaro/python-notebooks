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
# # Control Flow Tools

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## While loop
#
# - Don't forget the ':' character.
# - The body of the loop is indented

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.467648Z", "iopub.execute_input": "2020-09-12T13:29:14.468588Z", "iopub.status.idle": "2020-09-12T13:29:14.471122Z", "shell.execute_reply": "2020-09-12T13:29:14.471838Z"}}
# Fibonacci series:
# the sum of two elements defines the next
a, b = 0, 1
while b < 500:
    a, b = b, a+b
    print(round(b/a,5), end=",")

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## `if` Statements
#
# ```python
# True, False, and, or, not, ==, is, !=, is not, >, >=, <, <=
# ```
#
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.477066Z", "iopub.execute_input": "2020-09-12T13:29:14.477934Z", "iopub.status.idle": "2020-09-12T13:29:14.480455Z", "shell.execute_reply": "2020-09-12T13:29:14.481074Z"}}
x = 42
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# switch or case statements don't exist in Python.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise [Collatz conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture)
#
# Consider the following operation on an arbitrary positive integer:
#  - If the number is even, divide it by two.
#  - If the number is odd, triple it and add one.
#
# The conjecture is that no matter what initial value of this integer, the sequence will always reach 1.
#  - Test the Collatz conjecture for n = 100000.
#  - How many steps do you need to reach 1 ?
#  

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Loop over an iterable object
#
# We use for statement for looping over an iterable object. If we use it with a string, it loops over its characters.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.485077Z", "iopub.execute_input": "2020-09-12T13:29:14.485939Z", "iopub.status.idle": "2020-09-12T13:29:14.489057Z", "shell.execute_reply": "2020-09-12T13:29:14.488446Z"}}
for c in "python":
    print(c)

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.493411Z", "iopub.execute_input": "2020-09-12T13:29:14.494295Z", "iopub.status.idle": "2020-09-12T13:29:14.497306Z", "shell.execute_reply": "2020-09-12T13:29:14.496704Z"}}
for word in "Python ENSAI september 7th 2020".split(" "):
    print(word, len(word))   

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise: Anagram
#
# An anagram is word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
#
# Write a code that print True if s1 is an anagram of s2. 
# To do it, remove every character present in both strings. Check 
# you obtain two empty strings.
#
# Hint: `s = s.replace(c,"",1)` removes the character `c` in string `s` one time.
#
# ```python
# s1 = "pascal obispo"
# s2 = "pablo picasso"
# ..
# True
# ```

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Loop with range function
#
# - It generates arithmetic progressions
# - It is possible to let the range start at another number, or to specify a different increment.
# - Since Python 3, the object returned by `range()` doesn’t return a list to save memory space. `xrange` no longer exists.
# - Use function list() to creates it.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.509656Z", "iopub.execute_input": "2020-09-12T13:29:14.510931Z", "iopub.status.idle": "2020-09-12T13:29:14.513497Z", "shell.execute_reply": "2020-09-12T13:29:14.514125Z"}}
list(range(5))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.518878Z", "iopub.execute_input": "2020-09-12T13:29:14.519873Z", "iopub.status.idle": "2020-09-12T13:29:14.522334Z", "shell.execute_reply": "2020-09-12T13:29:14.523025Z"}}
list(range(2, 5))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.527604Z", "iopub.execute_input": "2020-09-12T13:29:14.528555Z", "iopub.status.idle": "2020-09-12T13:29:14.531825Z", "shell.execute_reply": "2020-09-12T13:29:14.531192Z"}}
list(range(-1, -5, -1))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.535974Z", "iopub.execute_input": "2020-09-12T13:29:14.536885Z", "iopub.status.idle": "2020-09-12T13:29:14.539202Z", "shell.execute_reply": "2020-09-12T13:29:14.539818Z"}}
for i in range(5):
    print(i, end=' ')

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise Exponential
#
# - Write some code to compute the exponential mathematical constant $e \simeq 2.718281828459045$ using the taylor series developed at 0 and without any import of external modules:
#
# $$ e \simeq \sum_{n=0}^{50} \frac{1}{n!} $$

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## `break` Statement.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.545429Z", "iopub.execute_input": "2020-09-12T13:29:14.546406Z", "iopub.status.idle": "2020-09-12T13:29:14.548856Z", "shell.execute_reply": "2020-09-12T13:29:14.549479Z"}}
for n in range(2, 10):     # n = 2,3,4,5,6,7,8,9
    for x in range(2, n):  # x = 2, ..., n-1
        if n % x == 0:     # Return the division remain (mod)
            print(n, " = ", x, "*", n//x)
            break
        else:
            print("%d is a prime number" % n)
            break

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ###  `iter` Function

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.554190Z", "iopub.execute_input": "2020-09-12T13:29:14.555184Z", "iopub.status.idle": "2020-09-12T13:29:14.557597Z", "shell.execute_reply": "2020-09-12T13:29:14.558257Z"}}
course = """ Python september 7, 14 2020 ENSAI Rennes """.split()
print(course)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.562257Z", "iopub.execute_input": "2020-09-12T13:29:14.563108Z", "iopub.status.idle": "2020-09-12T13:29:14.565344Z", "shell.execute_reply": "2020-09-12T13:29:14.565966Z"}}
iterator = iter(course)
print(iterator.__next__())

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.569788Z", "iopub.execute_input": "2020-09-12T13:29:14.570761Z", "iopub.status.idle": "2020-09-12T13:29:14.573163Z", "shell.execute_reply": "2020-09-12T13:29:14.573749Z"}}
print(iterator.__next__())


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Defining Function: `def` statement

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.578850Z", "iopub.execute_input": "2020-09-12T13:29:14.579774Z", "shell.execute_reply": "2020-09-12T13:29:14.582794Z", "iopub.status.idle": "2020-09-12T13:29:14.582210Z"}}
def is_palindromic(s):
    "Return True if the input sequence is a palindrome"
    return s == s[::-1]


is_palindromic("kayak")


# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - Body of the function start must be indented
# - Functions without a return statement do return a value called `None`.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.588088Z", "iopub.execute_input": "2020-09-12T13:29:14.589109Z", "iopub.status.idle": "2020-09-12T13:29:14.591652Z", "shell.execute_reply": "2020-09-12T13:29:14.592388Z"}}
def fib(n):
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
         print(a, end=' ')  # the end optional argument is \n by default
         a, b = b, a+b
    print("\n") # new line
     
result = fib(2000)
print(result) # is None


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Documentation string
# - It’s good practice to include docstrings in code that you write, so make a habit of it.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.596784Z", "iopub.execute_input": "2020-09-12T13:29:14.597757Z", "iopub.status.idle": "2020-09-12T13:29:14.599998Z", "shell.execute_reply": "2020-09-12T13:29:14.600637Z"}}
def my_function( foo):
     """Do nothing, but document it.

     No, really, it doesn't do anything.
     """
     pass

print(my_function.__doc__)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.605214Z", "iopub.execute_input": "2020-09-12T13:29:14.606173Z", "iopub.status.idle": "2020-09-12T13:29:14.608580Z", "shell.execute_reply": "2020-09-12T13:29:14.609204Z"}}
help(my_function)


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Default Argument Values

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.613940Z", "iopub.execute_input": "2020-09-12T13:29:14.614923Z", "iopub.status.idle": "2020-09-12T13:29:14.617292Z", "shell.execute_reply": "2020-09-12T13:29:14.617924Z"}}
def f(a,b=5):
    return a+b

print(f(1))
print(f(b="a",a="bc"))


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# **Important warning**: The default value is evaluated only once. 

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.622234Z", "iopub.execute_input": "2020-09-12T13:29:14.623182Z", "iopub.status.idle": "2020-09-12T13:29:14.625560Z", "shell.execute_reply": "2020-09-12T13:29:14.626220Z"}}
def f(a, L=[]):
    L.append(a)
    return L

print(f(1))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.630218Z", "iopub.execute_input": "2020-09-12T13:29:14.631229Z", "iopub.status.idle": "2020-09-12T13:29:14.633463Z", "shell.execute_reply": "2020-09-12T13:29:14.634043Z"}}
print(f(2)) # L = [1]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.637811Z", "iopub.execute_input": "2020-09-12T13:29:14.638655Z", "iopub.status.idle": "2020-09-12T13:29:14.640777Z", "shell.execute_reply": "2020-09-12T13:29:14.641393Z"}}
print(f(3)) # L = [1,2]


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Function Annotations
#
# Completely optional metadata information about the types used by user-defined functions.
# These type annotations conforming to [PEP 484](https://www.python.org/dev/peps/pep-0484/) could be statically used by [MyPy](http://mypy-lang.org).

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.646839Z", "iopub.execute_input": "2020-09-12T13:29:14.647791Z", "iopub.status.idle": "2020-09-12T13:29:14.650180Z", "shell.execute_reply": "2020-09-12T13:29:14.650799Z"}}
def f(ham: str, eggs: str = 'eggs') -> str:
     print("Annotations:", f.__annotations__)
     print("Arguments:", ham, eggs)
     return ham + ' and ' + eggs

f('spam')
help(f)
print(f.__doc__)


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Arbitrary Argument Lists
#
# Arguments can be wrapped up in a tuple or a list with form *args

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.655363Z", "iopub.execute_input": "2020-09-12T13:29:14.656254Z", "iopub.status.idle": "2020-09-12T13:29:14.658745Z", "shell.execute_reply": "2020-09-12T13:29:14.659364Z"}}
def f(*args, sep=" "):
    print (args)
    return sep.join(args)

print(f("big","data"))


# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - Normally, these variadic arguments will be last in the list of formal parameters. 
# - Any formal parameters which occur after the *args parameter are ‘keyword-only’ arguments.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Keyword Arguments Dictionary
#
# A final formal parameter of the form **name receives a dictionary.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.664054Z", "iopub.execute_input": "2020-09-12T13:29:14.664908Z", "iopub.status.idle": "2020-09-12T13:29:14.666369Z", "shell.execute_reply": "2020-09-12T13:29:14.667195Z"}}
def add_contact(kind, *args, **kwargs):
    print(args)
    print("-" * 40)
    for key, value in kwargs.items():
        print(key, ":", value)


# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# \*name must occur before \*\*name

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.671191Z", "iopub.execute_input": "2020-09-12T13:29:14.672042Z", "iopub.status.idle": "2020-09-12T13:29:14.674474Z", "shell.execute_reply": "2020-09-12T13:29:14.675093Z"}}
add_contact("John", "Smith",
           phone="555 8765",
           email="john.smith@python.org")

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Lambda Expressions
#
# Lambda functions can be used wherever function objects are required.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.679524Z", "iopub.execute_input": "2020-09-12T13:29:14.680542Z", "iopub.status.idle": "2020-09-12T13:29:14.683173Z", "shell.execute_reply": "2020-09-12T13:29:14.683772Z"}}
f = lambda x : 2 * x + 2
f(3)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.688394Z", "iopub.execute_input": "2020-09-12T13:29:14.689403Z", "shell.execute_reply": "2020-09-12T13:29:14.692349Z", "iopub.status.idle": "2020-09-12T13:29:14.691761Z"}}
taxicab_distance = lambda x_a,y_a,x_b,y_b: abs(x_b-x_a)+abs(y_b-y_a)
print(taxicab_distance(3,4,7,2))


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# lambda functions can reference variables from the containing scope:
#
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.697370Z", "iopub.execute_input": "2020-09-12T13:29:14.698323Z", "shell.execute_reply": "2020-09-12T13:29:14.701575Z", "iopub.status.idle": "2020-09-12T13:29:14.700970Z"}}
def make_incrementor(n):
    return lambda x: x + n

f = make_incrementor(42)
f(0),f(1)


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Unpacking Argument Lists
# Arguments are already in a list or tuple. They can be unpacked for a function call. 
# For instance, the built-in range() function is called with the *-operator to unpack the arguments out of a list:

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.707192Z", "iopub.execute_input": "2020-09-12T13:29:14.708137Z", "iopub.status.idle": "2020-09-12T13:29:14.710778Z", "shell.execute_reply": "2020-09-12T13:29:14.711390Z"}}
def chessboard_distance(x_a, y_a, x_b, y_b):
    """
    Compute the rectilinear distance between 
    point (x_a,y_a) and (x_b, y_b)
    """
    return max(abs(x_b-x_a),abs(y_b-y_a))

coordinates = [3,4,7,2] 
chessboard_distance(*coordinates)


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# In the same fashion, dictionaries can deliver keyword arguments with the **-operator:

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.716736Z", "iopub.execute_input": "2020-09-12T13:29:14.717694Z", "iopub.status.idle": "2020-09-12T13:29:14.720051Z", "shell.execute_reply": "2020-09-12T13:29:14.720709Z"}}
def parrot(voltage, state='a stiff', action='voom'):
     print("-- This parrot wouldn't", action, end=' ')
     print("if you put", voltage, "volts through it.", end=' ')
     print("E's", state, "!")

d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
parrot(**d)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Exercise: Time converter
# Write 3 functions to manipulate hours and minutes : 
# - Function minutes return minutes from (hours, minutes). 
# - Function hours the inverse function that return (hours, minutes) from minutes. 
# - Function add_time to add (hh1,mm1) and (hh2, mm2) two couples (hours, minutes). It takes 2
# tuples of length 2 as input arguments and return the tuple (hh,mm). 
#
# ```python
# print(minutes(6,15)) # 375 
# print(minutes(7,46)) # 466 
# print(add_time((6,15),(7,46)) # (14,01)
# ```

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Functions Scope
#
# - All variable assignments in a function store the value in the local symbol table.
# - Global variables cannot be directly assigned a value within a function (unless named in a global statement).
# - The value of the function can be assigned to another name which can then also be used as a function.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.725492Z", "iopub.execute_input": "2020-09-12T13:29:14.726483Z", "iopub.status.idle": "2020-09-12T13:29:14.728852Z", "shell.execute_reply": "2020-09-12T13:29:14.729458Z"}}
pi = 1.
def deg2rad(theta):
    pi = 3.14
    return theta * pi / 180.

print(deg2rad(45))
print(pi)


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.734168Z", "iopub.execute_input": "2020-09-12T13:29:14.735141Z", "iopub.status.idle": "2020-09-12T13:29:14.737497Z", "shell.execute_reply": "2020-09-12T13:29:14.738084Z"}}
def rad2deg(theta):
    return theta*180./pi

print(rad2deg(0.785))
pi = 3.14
print(rad2deg(0.785))


# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.742711Z", "iopub.execute_input": "2020-09-12T13:29:14.743727Z", "iopub.status.idle": "2020-09-12T13:29:14.746090Z", "shell.execute_reply": "2020-09-12T13:29:14.746694Z"}}
def deg2rad(theta):
    global pi
    pi = 3.14
    return theta * pi / 180

pi = 1
print(deg2rad(45))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.750485Z", "iopub.execute_input": "2020-09-12T13:29:14.751346Z", "iopub.status.idle": "2020-09-12T13:29:14.753722Z", "shell.execute_reply": "2020-09-12T13:29:14.754333Z"}}
print(pi)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## `enumerate` Function

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.759337Z", "iopub.execute_input": "2020-09-12T13:29:14.760310Z", "iopub.status.idle": "2020-09-12T13:29:14.762770Z", "shell.execute_reply": "2020-09-12T13:29:14.763400Z"}}
primes =  [1,2,3,5,7,11,13]
for idx, ele in enumerate (primes):
    print(idx, " --- ", ele) 

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise: Caesar cipher
#
# In cryptography, a Caesar cipher, is one of the simplest and most widely known encryption techniques. It is a type of substitution cipher in which each letter in the plaintext is replaced by a letter some fixed number of positions down the alphabet. For example, with a left shift of 3, D would be replaced by A, E would become B, and so on. 
#
# - Create a function `cipher` that take the plain text and the key value as arguments and return the encrypted text.
# - Create a funtion `plain` that take the crypted text and the key value as arguments that return the deciphered text.
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## `zip` Builtin Function
#
# Loop over sequences simultaneously.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.768763Z", "iopub.execute_input": "2020-09-12T13:29:14.769720Z", "shell.execute_reply": "2020-09-12T13:29:14.772732Z", "iopub.status.idle": "2020-09-12T13:29:14.772110Z"}}
L1 = [1, 2, 3]
L2 = [4, 5, 6]

for (x, y) in zip(L1, L2):
    print (x, y, '--', x + y)
# -

# <!-- #endregion -->

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## List comprehension
#
# - Set or change values inside a list
# - Create list from function

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.778120Z", "iopub.execute_input": "2020-09-12T13:29:14.779126Z", "iopub.status.idle": "2020-09-12T13:29:14.781794Z", "shell.execute_reply": "2020-09-12T13:29:14.782541Z"}}
lsingle = [1, 3, 9, 4]
ldouble = []
for k in lsingle:
    ldouble.append(2*k)
ldouble

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.786849Z", "iopub.execute_input": "2020-09-12T13:29:14.787771Z", "iopub.status.idle": "2020-09-12T13:29:14.789666Z", "shell.execute_reply": "2020-09-12T13:29:14.790297Z"}}
ldouble = [k*2 for k in lsingle]

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.795048Z", "iopub.execute_input": "2020-09-12T13:29:14.796013Z", "iopub.status.idle": "2020-09-12T13:29:14.798669Z", "shell.execute_reply": "2020-09-12T13:29:14.799275Z"}}
[n*n for n in range(1,10)]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.803997Z", "iopub.execute_input": "2020-09-12T13:29:14.804899Z", "iopub.status.idle": "2020-09-12T13:29:14.807474Z", "shell.execute_reply": "2020-09-12T13:29:14.808076Z"}}
[n*n for n in range(1,10) if n&1]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.813010Z", "iopub.execute_input": "2020-09-12T13:29:14.814006Z", "iopub.status.idle": "2020-09-12T13:29:14.816599Z", "shell.execute_reply": "2020-09-12T13:29:14.817212Z"}}
[n+1 if n&1 else n//2 for n in range(1,10) ]

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise
#
# Code a new version of cypher function using list comprehension. 
#
# Hints: 
# - `s = ''.join(L)` convert the characters list `L` into a string `s`.
# - `L.index(c)` return the index position of `c` in list `L` 
# - `"c".islower()` and `"C".isupper()` return `True`

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## `map` built-in function
#
# Apply a function over a sequence.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.821299Z", "iopub.execute_input": "2020-09-12T13:29:14.822218Z", "iopub.status.idle": "2020-09-12T13:29:14.824322Z", "shell.execute_reply": "2020-09-12T13:29:14.825007Z"}}
res = map(hex,range(16))
print(res)

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# Since Python 3.x, `map` process return an iterator. Save memory, and should make things go faster.
# Display result by using unpacking operator.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.829095Z", "iopub.execute_input": "2020-09-12T13:29:14.830042Z", "iopub.status.idle": "2020-09-12T13:29:14.832336Z", "shell.execute_reply": "2020-09-12T13:29:14.832957Z"}}
print(*res)


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## `map` with user-defined function

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.838098Z", "iopub.execute_input": "2020-09-12T13:29:14.838992Z", "iopub.status.idle": "2020-09-12T13:29:14.841455Z", "shell.execute_reply": "2020-09-12T13:29:14.842072Z"}}
def add(x,y):
    return x+y

L1 = [1, 2, 3]
L2 = [4, 5, 6]
print(*map(add,L1,L2))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - `map` is often faster than `for` loop

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:14.884017Z", "iopub.execute_input": "2020-09-12T13:29:14.931326Z", "iopub.status.idle": "2020-09-12T13:29:19.210242Z", "shell.execute_reply": "2020-09-12T13:29:19.210943Z"}}
M = range(10000)
f = lambda x: x**2
%timeit lmap = list(map(f,M))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:19.257945Z", "iopub.execute_input": "2020-09-12T13:29:19.300896Z", "iopub.status.idle": "2020-09-12T13:29:23.898271Z", "shell.execute_reply": "2020-09-12T13:29:23.898900Z"}}
M = range(10000)
f = lambda x: x**2
%timeit lfor = [f(m) for m in M]

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## filter
# creates a iterator of elements for which a function returns `True`. 

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:23.903570Z", "iopub.execute_input": "2020-09-12T13:29:23.904580Z", "iopub.status.idle": "2020-09-12T13:29:23.907100Z", "shell.execute_reply": "2020-09-12T13:29:23.907718Z"}}
number_list = range(-5, 5)
odd_numbers = filter(lambda x: x & 1 , number_list)
print(*odd_numbers)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - As `map`, `filter` is often faster than `for` loop

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:23.954183Z", "iopub.execute_input": "2020-09-12T13:29:23.996941Z", "iopub.status.idle": "2020-09-12T13:29:26.399288Z", "shell.execute_reply": "2020-09-12T13:29:26.399922Z"}}
M = range(1000)
f = lambda x: x % 3 == 0
%timeit lmap = filter(f,M)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:26.446338Z", "iopub.execute_input": "2020-09-12T13:29:26.490172Z", "iopub.status.idle": "2020-09-12T13:29:30.541223Z", "shell.execute_reply": "2020-09-12T13:29:30.541897Z"}}
M = range(1000)
%timeit lfor = (m for m in M if m % 3 == 0)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Exercise with map:
#
# Code a new version of your cypher function using map. 
#
# Hints: 
# - Applied function must have only one argument, create a function called `shift` with the key value and use map.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Exercise with filter:
#
# Create a function with a number n as single argument that returns True if n is a [Kaprekar number](https://en.wikipedia.org/wiki/Kaprekar_number). For example 45 is a Kaprekar number, because 
# $$45^2 = 2025$$ 
# and 
# $$20 + 25 = 45$$
#
# Use `filter` to give Kaprekar numbers list lower than 10000.
# ```
# 1, 9, 45, 55, 99, 297, 703, 999, 2223, 2728, 4879, 4950, 5050, 5292, 7272, 7777, 9999
# ```

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Recursive Call
#
# ```python slideshow={"slide_type": "fragment"}
# def gcd(x, y): 
#     """ returns the greatest common divisor."""
#     if x == 0: 
#         return y
#     else : 
#         return gcd(y % x, x)
#
# gcd(12,16)
# ```
# -

# ## Exercises

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Factorial
#
# - Write the function `factorial` with a recursive call
#
# NB: Recursion is not recommended by [Guido](http://neopythonic.blogspot.co.uk/2009/04/tail-recursion-elimination.html).

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Minimum number of rooms required for lectures.
#
# Given an array of time intervals (start, end) for classroom lectures (possibly overlapping), find the minimum number of rooms required.
#
# For example, given Input: 
# ```python
# lectures = ["9:00-10:30", "9:30-11:30","11:00-12:00","14:00-18:00", "15:00-16:00", "15:30-17:30", "16:00-18:00"]
# ```
# should output 3.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### [Non-palindromic skinny numbers](https://oeis.org/A035123)
#
# non-palindromic squares remaining square when written backwards
#
# $$
# \begin{array}{lclclcl}
# 10^2  &=& 100   &\qquad& 01^2  &=& 001 \\
# 13^2  &=& 169   &\qquad& 31^2  &=& 961 \\
# 102^2 &=& 10404 &\qquad& 201^2 &=& 40401
# \end{array}
# $$
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Narcissistic number
#
# A  number is narcissistic if the sum of its own digits each raised to the power of the number of digits. 
#
# Example : $4150 = 4^5 + 1^5 + 5^5 + 0^5$ or $153 = 1^3 + 5^3 + 3^3$
#
# Find narcissitic numbers with 3 digits
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Happy number
#
# - Given a number $n = n_0$, define a sequence $n_1, n_2,\ldots$ where 
#     $n_{{i+1}}$ is the sum of the squares of the digits of $n_{i}$. 
#     Then $n$ is happy if and only if there exists i such that $n_{i}=1$.
#
# For example, 19 is happy, as the associated sequence is:
# $$
# \begin{array}{ccccccl}
# 1^2 &+& 9^2 & &     &=& 82 \\
# 8^2 &+& 2^2 & &     &=& 68 \\
# 6^2 &+& 8^2 & &     &=& 100 \\
# 1^2 &+& 0^2 &+& 0^2 &=& 1
# \end{array}
# $$
# - Write a function `ishappy(n)` that returns True if `n` is happy.
# - Write a function `happy(n)` that returns a list with all happy numbers < $n$.
#
# ```python
# happy(100) = [1, 7, 10, 13, 19, 23, 28, 31, 32, 44, 49, 68, 70, 79, 82, 86, 91, 94, 97]
# ```

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Longuest increasing subsequence
#
# Given N elements, write a program that prints the length of the longuest increasing subsequence whose adjacent element difference is one.
#
# Examples:
# ```
# a = [3, 10, 3, 11, 4, 5, 6, 7, 8, 12]
# Output : 6
# Explanation: 3, 4, 5, 6, 7, 8 is the longest increasing subsequence whose adjacent element differs by one.
# ```
# ```
# Input : a = [6, 7, 8, 3, 4, 5, 9, 10]
# Output : 5
# Explanation: 6, 7, 8, 9, 10 is the longest increasing subsequence
# ```

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Polynomial derivative
# - A Polynomial is represented by a Python list of its coefficients.
#     [1,5,-4] => $1+5x-4x^2$
# - Write the function diff(P,n) that return the nth derivative Q
# - Don't use any external package 😉
# ```
# diff([3,2,1,5,7],2) = [2, 30, 84]
# diff([-6,5,-3,-4,3,-4],3) = [-24, 72, -240]
# ```
