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

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.785644Z", "iopub.execute_input": "2020-09-12T13:51:08.786614Z", "iopub.status.idle": "2020-09-12T13:51:08.788267Z", "shell.execute_reply": "2020-09-12T13:51:08.788888Z"}}
class Car:
    pass

mycar = Car()  # Create an empty car record

# Fill the fields of the record
mycar.brand = 'Peugeot'
mycar.model = 308
mycar.year = 2015

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.800837Z", "iopub.execute_input": "2020-09-12T13:51:08.801882Z", "iopub.status.idle": "2020-09-12T13:51:08.805281Z", "shell.execute_reply": "2020-09-12T13:51:08.805872Z"}}
mycar.__dict__

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## namedtuple

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.810030Z", "iopub.execute_input": "2020-09-12T13:51:08.811021Z", "iopub.status.idle": "2020-09-12T13:51:08.812590Z", "shell.execute_reply": "2020-09-12T13:51:08.813205Z"}}
from collections import namedtuple

Car = namedtuple('Car', 'brand, model, year')

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.817696Z", "iopub.execute_input": "2020-09-12T13:51:08.818621Z", "iopub.status.idle": "2020-09-12T13:51:08.821193Z", "shell.execute_reply": "2020-09-12T13:51:08.821813Z"}}
mycar = Car('Peugeot', 308, 2015)
mycar

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.825856Z", "iopub.execute_input": "2020-09-12T13:51:08.826836Z", "iopub.status.idle": "2020-09-12T13:51:08.829418Z", "shell.execute_reply": "2020-09-12T13:51:08.830029Z"}}
mycar.year

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.834648Z", "iopub.execute_input": "2020-09-12T13:51:08.835618Z", "iopub.status.idle": "2020-09-12T13:51:08.838217Z", "shell.execute_reply": "2020-09-12T13:51:08.838808Z"}}
# Like tuples, namedtuples are immutable:
import sys
try:
    mycar.model = 3008
except:
    print(sys.exc_info()[0])


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.844169Z", "iopub.execute_input": "2020-09-12T13:51:08.845057Z", "shell.execute_reply": "2020-09-12T13:51:08.846634Z", "iopub.status.idle": "2020-09-12T13:51:08.847290Z"}}
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


# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.851791Z", "iopub.execute_input": "2020-09-12T13:51:08.852654Z", "iopub.status.idle": "2020-09-12T13:51:08.855155Z", "shell.execute_reply": "2020-09-12T13:51:08.855770Z"}}
mycar = Car('Peugeot', 308, 2015) # Instance
print(f' {mycar.brand} {mycar.model} {mycar.year}')
print(f' {mycar.age()} years old')

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.860331Z", "iopub.execute_input": "2020-09-12T13:51:08.861290Z", "iopub.status.idle": "2020-09-12T13:51:08.863674Z", "shell.execute_reply": "2020-09-12T13:51:08.864368Z"}}
mycar.year = 2017
mycar.age()


# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# - `mycar` is an *instance* of Car Class.
# - `mycar.age()` is a *method* of `Car` instance `mycar`.
# - `brand` and `model` are attributes of `Car` instance `mycar`.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Convert method to attribute
#
# Use the `property` decorator 

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.869716Z", "iopub.execute_input": "2020-09-12T13:51:08.870718Z", "iopub.status.idle": "2020-09-12T13:51:08.872104Z", "shell.execute_reply": "2020-09-12T13:51:08.872718Z"}}
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


# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.876970Z", "iopub.execute_input": "2020-09-12T13:51:08.877938Z", "iopub.status.idle": "2020-09-12T13:51:08.880569Z", "shell.execute_reply": "2020-09-12T13:51:08.881171Z"}}
mycar = Car('Peugeot', 308, 2015)
mycar.age  # age can now be used as an attribute

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.885659Z", "iopub.execute_input": "2020-09-12T13:51:08.886635Z", "iopub.status.idle": "2020-09-12T13:51:08.889041Z", "shell.execute_reply": "2020-09-12T13:51:08.889636Z"}}
try:
    mycar.age = 3 # a protected attribute
except:
    print(sys.exc_info()[0])

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## The new Python 3.7 DataClass

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.894662Z", "iopub.execute_input": "2020-09-12T13:51:08.895529Z", "iopub.status.idle": "2020-09-12T13:51:08.899673Z", "shell.execute_reply": "2020-09-12T13:51:08.900288Z"}}
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


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.904661Z", "iopub.execute_input": "2020-09-12T13:51:08.905537Z", "iopub.status.idle": "2020-09-12T13:51:08.908195Z", "shell.execute_reply": "2020-09-12T13:51:08.908809Z"}}
mycar = Car('Peugeot', 308, 2015)
mycar

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.913290Z", "iopub.execute_input": "2020-09-12T13:51:08.914287Z", "shell.execute_reply": "2020-09-12T13:51:08.916981Z", "iopub.status.idle": "2020-09-12T13:51:08.917574Z"}}
myothercar = Car('BMW', "1 Series", 2009)
myothercar


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Method Overriding
# - Every Python classes has a `__repr__()` method used when you call `print()` function.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.923067Z", "iopub.execute_input": "2020-09-12T13:51:08.924068Z", "iopub.status.idle": "2020-09-12T13:51:08.925485Z", "shell.execute_reply": "2020-09-12T13:51:08.926095Z"}}
class Car:
    """Simple example class with method overriding """

    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year

    def __repr__(self):
        return f"{self.year} {self.brand} {self.model} {self.__class__.__name__}"


# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.930248Z", "iopub.execute_input": "2020-09-12T13:51:08.931177Z", "iopub.status.idle": "2020-09-12T13:51:08.933589Z", "shell.execute_reply": "2020-09-12T13:51:08.934192Z"}}
mycar = Car('Peugeot', 308, 2015)
print(mycar)


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Inheritance

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.941271Z", "iopub.execute_input": "2020-09-12T13:51:08.942264Z", "iopub.status.idle": "2020-09-12T13:51:08.944691Z", "shell.execute_reply": "2020-09-12T13:51:08.945271Z"}}
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

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.949970Z", "iopub.execute_input": "2020-09-12T13:51:08.950830Z", "iopub.status.idle": "2020-09-12T13:51:08.952487Z", "shell.execute_reply": "2020-09-12T13:51:08.953140Z"}}
class DemoClass:
    " Demo class for name mangling "

    def public_method(self):
        return 'public!'

    def __private_method(self):  # Note the use of leading underscores
        return 'private!'


object3 = DemoClass()

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.957241Z", "iopub.execute_input": "2020-09-12T13:51:08.958219Z", "iopub.status.idle": "2020-09-12T13:51:08.960789Z", "shell.execute_reply": "2020-09-12T13:51:08.961388Z"}}
object3.public_method()

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.965615Z", "iopub.execute_input": "2020-09-12T13:51:08.966464Z", "iopub.status.idle": "2020-09-12T13:51:08.968701Z", "shell.execute_reply": "2020-09-12T13:51:08.969316Z"}}
try:
    object3.__private_method()
except:
    print(sys.exc_info()[0])

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.973812Z", "iopub.execute_input": "2020-09-12T13:51:08.974819Z", "iopub.status.idle": "2020-09-12T13:51:08.977314Z", "shell.execute_reply": "2020-09-12T13:51:08.978031Z"}}
[ s for s in dir(object3) if "method" in s]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.982303Z", "iopub.execute_input": "2020-09-12T13:51:08.983268Z", "iopub.status.idle": "2020-09-12T13:51:08.985636Z", "shell.execute_reply": "2020-09-12T13:51:08.986486Z"}}
object3._DemoClass__private_method()

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:08.990527Z", "iopub.execute_input": "2020-09-12T13:51:08.991673Z", "iopub.status.idle": "2020-09-12T13:51:08.994231Z", "shell.execute_reply": "2020-09-12T13:51:08.994828Z"}}
object3.public_method


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Use `class` as a Function.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.001329Z", "iopub.execute_input": "2020-09-12T13:51:09.002688Z", "iopub.status.idle": "2020-09-12T13:51:09.005389Z", "shell.execute_reply": "2020-09-12T13:51:09.005979Z"}}
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

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.016501Z", "iopub.execute_input": "2020-09-12T13:51:09.017415Z", "iopub.status.idle": "2020-09-12T13:51:09.019171Z", "shell.execute_reply": "2020-09-12T13:51:09.019827Z"}}
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


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.024309Z", "iopub.execute_input": "2020-09-12T13:51:09.025290Z", "iopub.status.idle": "2020-09-12T13:51:09.027854Z", "shell.execute_reply": "2020-09-12T13:51:09.028530Z"}}
u = MyComplex(0, 1)
v = MyComplex(1, 0)
print('u=', u, "; v=", v)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.033254Z", "iopub.execute_input": "2020-09-12T13:51:09.034193Z", "iopub.status.idle": "2020-09-12T13:51:09.036716Z", "shell.execute_reply": "2020-09-12T13:51:09.037300Z"}}
u+v, u-v, u*v, u==v

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# We can change the *class* attribute precision.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.041555Z", "iopub.execute_input": "2020-09-12T13:51:09.042409Z", "iopub.status.idle": "2020-09-12T13:51:09.044973Z", "shell.execute_reply": "2020-09-12T13:51:09.045586Z"}}
MyComplex.precision=2
print(u.precision)
print(u)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.049791Z", "iopub.execute_input": "2020-09-12T13:51:09.050862Z", "iopub.status.idle": "2020-09-12T13:51:09.053410Z", "shell.execute_reply": "2020-09-12T13:51:09.054051Z"}}
v.precision

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# We can change the *instance* attribute precision.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.058144Z", "iopub.execute_input": "2020-09-12T13:51:09.059063Z", "iopub.status.idle": "2020-09-12T13:51:09.061352Z", "shell.execute_reply": "2020-09-12T13:51:09.061952Z"}}
u.precision=1
print(u)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.065739Z", "iopub.execute_input": "2020-09-12T13:51:09.066740Z", "iopub.status.idle": "2020-09-12T13:51:09.069169Z", "shell.execute_reply": "2020-09-12T13:51:09.069934Z"}}
print(v)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.074115Z", "iopub.execute_input": "2020-09-12T13:51:09.075066Z", "iopub.status.idle": "2020-09-12T13:51:09.077687Z", "shell.execute_reply": "2020-09-12T13:51:09.078367Z"}}
MyComplex.precision=5
u # set attribute keeps its value

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.082453Z", "iopub.execute_input": "2020-09-12T13:51:09.083449Z", "iopub.status.idle": "2020-09-12T13:51:09.086044Z", "shell.execute_reply": "2020-09-12T13:51:09.086631Z"}}
v # unset attribute is set to the new value


# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Rational example

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.096857Z", "iopub.execute_input": "2020-09-12T13:51:09.097866Z", "iopub.status.idle": "2020-09-12T13:51:09.099409Z", "shell.execute_reply": "2020-09-12T13:51:09.100023Z"}}
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


# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:51:09.105230Z", "iopub.execute_input": "2020-09-12T13:51:09.106156Z", "iopub.status.idle": "2020-09-12T13:51:09.108659Z", "shell.execute_reply": "2020-09-12T13:51:09.109276Z"}}
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
