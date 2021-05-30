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
# # Classes
# - Classes provide a means of bundling data and functionality together.
# - Creating a new class creates a **new type** of object.
# - Assigned variables are new **instances** of that type.
# - Each class instance can have **attributes** attached to it.
# - Class instances can also have **methods** for modifying its state.
# - Python classes provide the class **inheritance** mechanism.
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Use class to store data
#
# - A empty class can be used to bundle together a few named data items. 
# - You can easily save this class containing your data in JSON file.

# + {"slideshow": {"slide_type": "fragment"}}
class Car:
    pass

mycar = Car()  # Create an empty car record

# Fill the fields of the record
mycar.brand = 'Peugeot'
mycar.model = 308
mycar.year = 2015

# + {"slideshow": {"slide_type": "fragment"}}
mycar.__dict__

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## namedtuple

# + {"slideshow": {"slide_type": "fragment"}}
from collections import namedtuple

Car = namedtuple('Car', 'brand, model, year')

# + {"slideshow": {"slide_type": "fragment"}}
mycar = Car('Peugeot', 308, 2015)
mycar

# + {"slideshow": {"slide_type": "fragment"}}
mycar.year

# + {"slideshow": {"slide_type": "fragment"}}
# Like tuples, namedtuples are immutable:
import sys
try:
    mycar.model = 3008
except:
    print(sys.exc_info()[0])


# + {"slideshow": {"slide_type": "slide"}}
class Car:

    "A simple example class Animal with its name, weight and age"

    def __init__(self, brand, model, year):  # constructor
        self.brand = brand
        self.model = model
        self.year = year

    def age(self):  # method
        import datetime
        now = datetime.datetime.now()
        return now.year - self.year


# + {"slideshow": {"slide_type": "fragment"}}
mycar = Car('Peugeot', 308, 2015) # Instance
print(f' {mycar.brand} {mycar.model} {mycar.year}')
print(f' {mycar.age()} years old')

# + {"slideshow": {"slide_type": "slide"}}
mycar.year = 2017
mycar.age()


# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - `mycar` is an *instance* of Car Class.
# - `mycar.age()` is a *method* of `Car` instance `mycar`.
# - `brand` and `model` are attributes of `Car` instance `mycar`.
# -

# ### Exercise 
# - Create the class `TennisPlaye`
# - Add 6 attributes that represent skills of this player : `agility`, `stamina`, `serve`, `volley`, `forehand`, `backhand`. Each quality is quantified by an integer.
# - Add the method `total_power`that computes the sum of all skills. You can use the `__dict__` method.
#
# ```py
# jonah = TennisPlayer( 16, 9, 2, 9, 15, 12)
# jonah.total_power() # must be 63
# ```

class TennisPlayer:

    def __init__(self, agility, stamina, serve, volley, forehand, backhand):
        self.agility = agility
        self.stamina = stamina
        self.serve = serve
        self.volley = volley
        self.forehand = forehand
        self.backhand = backhand

    def total_power(self):
        return sum(self.__dict__.values())
    def __repr__(self):
        s = f"AGILITY  \t {self.agility} \n"
        s += f"STAMINA  \t {self.stamina} \n"
        s += f"SERVE    \t {self.serve} \n"
        s += f"VOLLEY   \t {self.volley} \n"
        s += f"FOREHAND \t {self.forehand} \n"
        s += f"BACKHAND \t {self.backhand} \n"
        s += "------------------- \n"
        s += f" \t \t {self.total_power()} "
        return s


jonah = TennisPlayer( 16, 9, 2, 9, 15, 12)
jonah.total_power() # must be 63

sum(jonah.__dict__.values())


jonah


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Convert method to attribute
#
# Use the `property` decorator 

# + {"slideshow": {"slide_type": "fragment"}}
class Car:

    "A simple example class Car with its model, brand and year"

    def __init__(self, brand, model, year):  # constructor
        self.model = model
        self.brand = brand
        self.year = year

    @property
    def age(self):  # method
        import datetime
        now = datetime.datetime.now()
        return now.year - self.year


# + {"slideshow": {"slide_type": "fragment"}}
mycar = Car('Peugeot', 308, 2015)
mycar.age  # age can now be used as an attribute

# + {"slideshow": {"slide_type": "fragment"}}
try:
    mycar.age = 3 # a protected attribute
except:
    print(sys.exc_info()[0])

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## The new Python 3.7 DataClass

# + {"slideshow": {"slide_type": "fragment"}}
from dataclasses import dataclass

@dataclass
class Car:

    brand: str
    model: int
    year: int

    @property
    def age(self) -> int:
        import datetime
        now = datetime.datetime.now()
        return now.year - self.year


# + {"slideshow": {"slide_type": "slide"}}
mycar = Car('Peugeot', 308, 2015)
mycar

# + {"slideshow": {"slide_type": "fragment"}}
myothercar = Car('BMW', "1 Series", 2009)
myothercar


# -

# ### Exercise
#
# Simplify the `TennisPlayer` class by using `dataclass`.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Method Overriding
# - Every Python classes has a `__repr__()` method used when you call `print()` function.

# + {"slideshow": {"slide_type": "fragment"}}
class Car:
    """Simple example class with method overriding """

    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year

    def __repr__(self):
        return f"{self.year} {self.brand} {self.model} {self.__class__.__name__}"


# + {"slideshow": {"slide_type": "fragment"}}
mycar = Car('Peugeot', 308, 2015)
print(mycar)


# -

# ### Exercise
#
# - Add a representation method to your `TennisPlayer` class 
# ```py
# print(jonah)
# ```
# ~~~
# AGILITY  	 16 
# STAMINA  	 9 
# SERVE    	 2 
# VOLLEY   	 9 
# FOREHAND 	 15 
# BACKHAND 	 12
# ~~~

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Inheritance

# + {"slideshow": {"slide_type": "fragment"}}
class Rectangle():  # Parent class is defined here

    def __init__(self, width, height):
        self.width, self.height = width, height
    @property
    def area(self):
        return self.width * self.height
    
class Square(Rectangle):
    
    def __init__(self, edge):
        super().__init__(edge, edge)  # Call method in the parent class


r = Rectangle(2, 3)
print(f"Rectangle area \t = {r.area:7.3f}")
s = Square(4)
print(f"Square area \t = {s.area:7.3f}")


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Private Variables and Methods

# + {"slideshow": {"slide_type": "fragment"}}
class DemoClass:
    " Demo class for name mangling "

    def public_method(self):
        return 'public!'

    def __private_method(self):  # Note the use of leading underscores
        return 'private!'


object3 = DemoClass()

# + {"slideshow": {"slide_type": "slide"}}
object3.public_method()

# + {"slideshow": {"slide_type": "fragment"}}
try:
    object3.__private_method()
except:
    print(sys.exc_info()[0])

# + {"slideshow": {"slide_type": "fragment"}}
[ s for s in dir(object3) if "method" in s]

# + {"slideshow": {"slide_type": "fragment"}}
object3._DemoClass__private_method()

# + {"slideshow": {"slide_type": "fragment"}}
object3.public_method


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Use `class` as a Function.

# + {"slideshow": {"slide_type": "fragment"}}
class Polynomial:
    
   " Class representing a polynom P(x) -> c_0+c_1*x+c_2*x^2+..."
    
   def __init__(self, coeffs):
      self.coeffs = coeffs
        
   def __call__(self, x):
      return sum(coef*x**exp for exp,coef in enumerate(self.coeffs))

p = Polynomial([2,4,-1])
p(2) 


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise: Polynomial
#
# - Improve the class above called Polynomial by creating a method `diff(n)` to compute the nth derivative.
# - Override the `__repr__()` method to output a pretty printing.
#
# Hint: `f"{coeff:+d}"` forces to print sign before the value of an integer.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Operators Overriding 

# + {"slideshow": {"slide_type": "slide"}}
class MyComplex:
    " Simple class representing a complex"
    width = 7
    precision = 3

    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def __repr__(self): 
        return (f"({self.real:{self.width}.{self.precision}f},"
                f"{self.imag:+{self.width}.{self.precision}f}j)")

    def __eq__(self, other):  # override '=='
        return (self.real == other.real) and (self.imag == other.imag)

    def __add__(self, other):  # override '+'
        return MyComplex(self.real+other.real, self.imag+other.imag)

    def __sub__(self, other):  # override '-'
        return MyComplex(self.real-other.real, self.imag-other.imag)

    def __mul__(self, other):  # override '*'
        if isinstance(other, MyComplex):
            return MyComplex(self.real * other.real - self.imag * other.imag,
                             self.real * other.imag + self.imag * other.real)

        else:
            return MyComplex(other*self.real, other*self.imag)


# + {"slideshow": {"slide_type": "slide"}}
u = MyComplex(0, 1)
v = MyComplex(1, 0)
print('u=', u, "; v=", v)

# + {"slideshow": {"slide_type": "fragment"}}
u+v, u-v, u*v, u==v

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# We can change the *class* attribute precision.

# + {"slideshow": {"slide_type": "fragment"}}
MyComplex.precision=2
print(u.precision)
print(u)

# + {"slideshow": {"slide_type": "fragment"}}
v.precision

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# We can change the *instance* attribute precision.

# + {"slideshow": {"slide_type": "fragment"}}
u.precision=1
print(u)

# + {"slideshow": {"slide_type": "fragment"}}
print(v)

# + {"slideshow": {"slide_type": "fragment"}}
MyComplex.precision=5
u # set attribute keeps its value

# + {"slideshow": {"slide_type": "fragment"}}
v # unset attribute is set to the new value


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Rational example

# + {"slideshow": {"slide_type": "slide"}}
class Rational:
    " Class representing a rational number"

    def __init__(self, n, d):
        assert isinstance(n, int) and isinstance(d, int)

        def gcd(x, y):
            if x == 0:
                return y
            elif x < 0:
                return gcd(-x, y)
            elif y < 0:
                return -gcd(x, -y)
            else:
                return gcd(y % x, x)

        g = gcd(n, d)
        self.numer, self.denom = n//g, d//g

    def __add__(self, other):
        return Rational(self.numer * other.denom + other.numer * self.denom,
                        self.denom * other.denom)

    def __sub__(self, other):
        return Rational(self.numer * other.denom - other.numer * self.denom,
                        self.denom * other.denom)

    def __mul__(self, other):
        return Rational(self.numer * other.numer, self.denom * other.denom)

    def __truediv__(self, other):
        return Rational(self.numer * other.denom, self.denom * other.numer)

    def __repr__(self):
        return f"{self.numer:d}/{self.denom:d}"


# + {"slideshow": {"slide_type": "slide"}}
r1 = Rational(2,3)
r2 = Rational(3,4)
r1+r2, r1-r2, r1*r2, r1/r2

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise 
# Improve the class Polynomial by implementing operations:
# - Overrides '+' operator (__add__)
# - Overrides '-' operator (__neg__)
# - Overrides '==' operator (__eq__)
# - Overrides '*' operator (__mul__)
