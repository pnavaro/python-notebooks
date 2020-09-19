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
# # Iterators
# Most container objects can be looped over using a for statement:

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.229808Z", "iopub.execute_input": "2020-09-12T14:00:01.231861Z", "shell.execute_reply": "2020-09-12T14:00:01.234743Z", "iopub.status.idle": "2020-09-12T14:00:01.235347Z"}}
for element in [1, 2, 3]:
    print(element, end=' ')

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.239599Z", "iopub.execute_input": "2020-09-12T14:00:01.240524Z", "iopub.status.idle": "2020-09-12T14:00:01.242852Z", "shell.execute_reply": "2020-09-12T14:00:01.243426Z"}}
for element in (1, 2, 3):
    print(element, end=' ')

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.247851Z", "iopub.execute_input": "2020-09-12T14:00:01.248718Z", "iopub.status.idle": "2020-09-12T14:00:01.250913Z", "shell.execute_reply": "2020-09-12T14:00:01.251475Z"}}
for key in {'one': 1, 'two': 2}:
    print(key, end=' ')

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.255572Z", "iopub.execute_input": "2020-09-12T14:00:01.256381Z", "iopub.status.idle": "2020-09-12T14:00:01.258449Z", "shell.execute_reply": "2020-09-12T14:00:01.259032Z"}}
for char in "123":
    print(char, end=' ')

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.263330Z", "iopub.execute_input": "2020-09-12T14:00:01.264205Z", "iopub.status.idle": "2020-09-12T14:00:01.266965Z", "shell.execute_reply": "2020-09-12T14:00:01.267547Z"}}
for line in open("../environment.yml"):
    print(line, end= ' ')

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - The `for` statement calls `iter()` on the container object. 
# - The function returns an iterator object that defines the method `__next__()`
# - To add iterator behavior to your classes: 
#     - Define an `__iter__()` method which returns an object with a `__next__()`.
#     - If the class defines `__next__()`, then `__iter__()` can just return self.
#     - The **StopIteration** exception indicates the end of the loop.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.280461Z", "iopub.execute_input": "2020-09-12T14:00:01.281505Z", "iopub.status.idle": "2020-09-12T14:00:01.285011Z", "shell.execute_reply": "2020-09-12T14:00:01.285587Z"}}
s = 'abc'
it = iter(s)
it

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.290652Z", "iopub.execute_input": "2020-09-12T14:00:01.291631Z", "iopub.status.idle": "2020-09-12T14:00:01.294202Z", "shell.execute_reply": "2020-09-12T14:00:01.294776Z"}}
next(it), next(it), next(it)


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.301316Z", "iopub.execute_input": "2020-09-12T14:00:01.302273Z", "iopub.status.idle": "2020-09-12T14:00:01.303519Z", "shell.execute_reply": "2020-09-12T14:00:01.304316Z"}}
class Reverse:
    """Iterator for looping over a sequence backwards."""

    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]


# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.308328Z", "iopub.execute_input": "2020-09-12T14:00:01.309270Z", "iopub.status.idle": "2020-09-12T14:00:01.311597Z", "shell.execute_reply": "2020-09-12T14:00:01.312173Z"}}
rev = Reverse('spam')
for char in rev:
    print(char, end='')


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.317364Z", "iopub.execute_input": "2020-09-12T14:00:01.318250Z", "iopub.status.idle": "2020-09-12T14:00:01.320548Z", "shell.execute_reply": "2020-09-12T14:00:01.321123Z"}}
def reverse(data): # Python 3.6
    yield from data[::-1]
    
for char in reverse('bulgroz'):
     print(char, end='')


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Generators
# - Generators are a simple and powerful tool for creating iterators.
# - Write regular functions but use the yield statement when you want to return data.
# - the `__iter__()` and `__next__()` methods are created automatically.
#

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.325694Z", "iopub.execute_input": "2020-09-12T14:00:01.326496Z", "iopub.status.idle": "2020-09-12T14:00:01.328060Z", "shell.execute_reply": "2020-09-12T14:00:01.328721Z"}}
def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]


# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.332867Z", "iopub.execute_input": "2020-09-12T14:00:01.333750Z", "iopub.status.idle": "2020-09-12T14:00:01.335915Z", "shell.execute_reply": "2020-09-12T14:00:01.336509Z"}}
for char in reverse('bulgroz'):
     print(char, end='')
# -

# ### Exercise 
#
# Generates a list of IP addresses based on IP range. 
#
# ```python
# ip_range = 
# for ip in ip_range("192.168.1.0", "192.168.1.10"):
#    print(ip)
#
# 192.168.1.0
# 192.168.1.1
# 192.168.1.2
# ...
# ```

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Generator Expressions
#
# - Use a syntax similar to list comprehensions but with parentheses instead of brackets.
# - Tend to be more memory friendly than equivalent list comprehensions.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.340870Z", "iopub.execute_input": "2020-09-12T14:00:01.341775Z", "iopub.status.idle": "2020-09-12T14:00:01.344138Z", "shell.execute_reply": "2020-09-12T14:00:01.344710Z"}}
sum(i*i for i in range(10))                 # sum of squares

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.349086Z", "iopub.execute_input": "2020-09-12T14:00:01.349916Z", "iopub.status.idle": "2020-09-12T14:00:01.881615Z", "shell.execute_reply": "2020-09-12T14:00:01.882538Z"}}
%load_ext memory_profiler

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:01.911881Z", "iopub.execute_input": "2020-09-12T14:00:01.912654Z", "iopub.status.idle": "2020-09-12T14:00:02.099114Z", "shell.execute_reply": "2020-09-12T14:00:02.099705Z"}}
%memit doubles = [2 * n for n in range(10000)]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:02.133082Z", "iopub.execute_input": "2020-09-12T14:00:02.133855Z", "iopub.status.idle": "2020-09-12T14:00:02.307287Z", "shell.execute_reply": "2020-09-12T14:00:02.307891Z"}}
%memit doubles = (2 * n for n in range(10000))

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:02.313384Z", "iopub.execute_input": "2020-09-12T14:00:02.314318Z", "shell.execute_reply": "2020-09-12T14:00:02.316635Z", "iopub.status.idle": "2020-09-12T14:00:02.317286Z"}}
# list comprehension
doubles = [2 * n for n in range(10)]
for x in doubles:
    print(x, end=' ')

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:02.322430Z", "iopub.execute_input": "2020-09-12T14:00:02.323362Z", "iopub.status.idle": "2020-09-12T14:00:02.325630Z", "shell.execute_reply": "2020-09-12T14:00:02.326203Z"}}
# generator expression
doubles = (2 * n for n in range(10))
for x in doubles:
    print(x, end=' ')

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise
#
# The [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) of the first kind are defined by the recurrence relation
#
# \begin{align}
# T_o(x) &= 1 \\
# T_1(x) &= x \\
# T_{n+1} &= 2xT_n(x)-T_{n-1}(x)
# \end{align}
#
# - Create a class `Chebyshev` that generates the sequence of Chebyshev polynomials

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## itertools
#
# ### zip_longest
#
# `itertools.zip_longest()` accepts any number of iterables 
# as arguments and a fillvalue keyword argument that defaults to None.
#     

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:02.332731Z", "iopub.execute_input": "2020-09-12T14:00:02.333626Z", "iopub.status.idle": "2020-09-12T14:00:02.336113Z", "shell.execute_reply": "2020-09-12T14:00:02.336685Z"}}
x = [1, 1, 1, 1, 1]
y = [1, 2, 3, 4, 5, 6, 7]
list(zip(x, y))
from itertools import zip_longest
list(map(sum,zip_longest(x, y, fillvalue=1)))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### combinations

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:02.340533Z", "iopub.execute_input": "2020-09-12T14:00:02.341429Z", "iopub.status.idle": "2020-09-12T14:00:02.342916Z", "shell.execute_reply": "2020-09-12T14:00:02.343486Z"}}
loto_numbers = list(range(1,50))

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# A choice of 6 numbers from the sequence  1 to 49 is called a combination. 
# The `itertools.combinations()` function takes two arguments—an iterable 
# inputs and a positive integer n—and produces an iterator over tuples of 
# all combinations of n elements in inputs.

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:05.337335Z", "iopub.execute_input": "2020-09-12T14:00:05.338237Z", "iopub.status.idle": "2020-09-12T14:00:05.340647Z", "shell.execute_reply": "2020-09-12T14:00:05.341222Z"}}
from itertools import combinations
len(list(combinations(loto_numbers, 6)))

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:05.345919Z", "iopub.execute_input": "2020-09-12T14:00:05.346879Z", "iopub.status.idle": "2020-09-12T14:00:05.349205Z", "shell.execute_reply": "2020-09-12T14:00:05.349768Z"}}
from math import factorial
factorial(49)/ factorial(6) / factorial(49-6)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### permutations

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:05.353904Z", "iopub.execute_input": "2020-09-12T14:00:05.354737Z", "iopub.status.idle": "2020-09-12T14:00:05.357111Z", "shell.execute_reply": "2020-09-12T14:00:05.357706Z"}}
from itertools import permutations
for s in permutations('dsi'):
    print( "".join(s), end=", ")

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### count

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:05.363006Z", "iopub.execute_input": "2020-09-12T14:00:05.363820Z", "iopub.status.idle": "2020-09-12T14:00:05.365945Z", "shell.execute_reply": "2020-09-12T14:00:05.366514Z"}}
from itertools import count
n = 2024
for k in count(): # replace  k = 0; while(True) : k += 1
    if n == 1:
        print(f"k = {k}")
        break
    elif n & 1:
        n = 3*n +1
    else:
        n = n // 2


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### cycle, islice, dropwhile, takewhile

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:05.372223Z", "iopub.execute_input": "2020-09-12T14:00:05.373125Z", "iopub.status.idle": "2020-09-12T14:00:05.375339Z", "shell.execute_reply": "2020-09-12T14:00:05.375919Z"}}
from itertools import cycle, islice, dropwhile, takewhile
L = list(range(10))
cycled = cycle(L)  # cycle through the list 'L'
skipped = dropwhile(lambda x: x < 6 , cycled)  # drop the values until x==4
sliced = islice(skipped, None, 20)  # take the first 20 values
print(*sliced)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:05.380312Z", "iopub.execute_input": "2020-09-12T14:00:05.381246Z", "iopub.status.idle": "2020-09-12T14:00:05.383417Z", "shell.execute_reply": "2020-09-12T14:00:05.383992Z"}}
result = takewhile(lambda x: x > 0, cycled) # cycled begins to 4
print(*result)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### product

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T14:00:05.390668Z", "iopub.execute_input": "2020-09-12T14:00:05.391591Z", "iopub.status.idle": "2020-09-12T14:00:05.394021Z", "shell.execute_reply": "2020-09-12T14:00:05.394598Z"}}
ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7']
suits = [ '\u2660', '\u2665', '\u2663', '\u2666']
cards = [(rank, suit) for rank in ranks for suit in suits]
len(cards)
from itertools import product
cards = product(ranks, suits)
print(*cards)
