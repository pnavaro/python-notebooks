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
# # Iterators
# Most container objects can be looped over using a for statement:

# + {"slideshow": {"slide_type": "slide"}}
for element in [1, 2, 3]:
    print(element, end=' ')

# + {"slideshow": {"slide_type": "slide"}}
for element in (1, 2, 3):
    print(element, end=' ')

# + {"slideshow": {"slide_type": "slide"}}
for key in {'one': 1, 'two': 2}:
    print(key, end=' ')

# + {"slideshow": {"slide_type": "slide"}}
for char in "123":
    print(char, end=' ')

# + {"slideshow": {"slide_type": "slide"}}
for line in open("../environment.yml"):
    print(line, end= ' ')

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - The `for` statement calls `iter()` on the container object. 
# - The function returns an iterator object that defines the method `__next__()`
# - To add iterator behavior to your classes: 
#     - Define an `__iter__()` method which returns an object with a `__next__()`.
#     - If the class defines `__next__()`, then `__iter__()` can just return self.
#     - The **StopIteration** exception indicates the end of the loop.

# + {"slideshow": {"slide_type": "fragment"}}
s = 'abc'
it = iter(s)
it

# + {"slideshow": {"slide_type": "fragment"}}
next(it), next(it), next(it)


# + {"slideshow": {"slide_type": "slide"}}
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


# + {"slideshow": {"slide_type": "fragment"}}
rev = Reverse('spam')
for char in rev:
    print(char, end='')


# + {"slideshow": {"slide_type": "slide"}}
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

# + {"slideshow": {"slide_type": "fragment"}}
def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]


# + {"slideshow": {"slide_type": "fragment"}}
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

# + {"slideshow": {"slide_type": "fragment"}}
sum(i*i for i in range(10))                 # sum of squares

# + {"slideshow": {"slide_type": "fragment"}}
%load_ext memory_profiler

# + {"slideshow": {"slide_type": "fragment"}}
%memit doubles = [2 * n for n in range(10000)]

# + {"slideshow": {"slide_type": "fragment"}}
%memit doubles = (2 * n for n in range(10000))

# + {"slideshow": {"slide_type": "slide"}}
# list comprehension
doubles = [2 * n for n in range(10)]
for x in doubles:
    print(x, end=' ')

# + {"slideshow": {"slide_type": "fragment"}}
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

# + {"slideshow": {"slide_type": "fragment"}}
x = [1, 1, 1, 1, 1]
y = [1, 2, 3, 4, 5, 6, 7]
list(zip(x, y))
from itertools import zip_longest
list(map(sum,zip_longest(x, y, fillvalue=1)))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### combinations

# + {"slideshow": {"slide_type": "fragment"}}
loto_numbers = list(range(1,50))

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# A choice of 6 numbers from the sequence  1 to 49 is called a combination. 
# The `itertools.combinations()` function takes two arguments—an iterable 
# inputs and a positive integer n—and produces an iterator over tuples of 
# all combinations of n elements in inputs.

# + {"slideshow": {"slide_type": "slide"}}
from itertools import combinations
len(list(combinations(loto_numbers, 6)))

# + {"slideshow": {"slide_type": "fragment"}}
from math import factorial
factorial(49)/ factorial(6) / factorial(49-6)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### permutations

# + {"slideshow": {"slide_type": "fragment"}}
from itertools import permutations
for s in permutations('dsi'):
    print( "".join(s), end=", ")

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### count

# + {"slideshow": {"slide_type": "fragment"}}
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

# + {"slideshow": {"slide_type": "fragment"}}
from itertools import cycle, islice, dropwhile, takewhile
L = list(range(10))
cycled = cycle(L)  # cycle through the list 'L'
skipped = dropwhile(lambda x: x < 6 , cycled)  # drop the values until x==4
sliced = islice(skipped, None, 20)  # take the first 20 values
print(*sliced)

# + {"slideshow": {"slide_type": "fragment"}}
result = takewhile(lambda x: x > 0, cycled) # cycled begins to 4
print(*result)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### product

# + {"slideshow": {"slide_type": "fragment"}}
ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7']
suits = [ '\u2660', '\u2665', '\u2663', '\u2666']
cards = [(rank, suit) for rank in ranks for suit in suits]
len(cards)
from itertools import product
cards = product(ranks, suits)
print(*cards)
