---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region slideshow={"slide_type": "slide"} -->
# Errors and Exceptions

There are two distinguishable kinds of errors: *syntax errors* and *exceptions*.
- Syntax errors, also known as parsing errors, are the most common.
- Exceptions are errors caused by statement or expression syntactically corrects.
- Exceptions are not unconditionally fatal.

[Exceptions in Python documentation](https://docs.python.org/3/library/exceptions.html)
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
import sys
try:
    10 * (1/0)
except:
    print(sys.exc_info()[0])
```

```python slideshow={"slide_type": "slide"}
try:
    4 + spam*3
except:
    print(sys.exc_info()[0])
```

```python slideshow={"slide_type": "slide"}
try:
    '2' + 2
except:
    print(sys.exc_info()[0])
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Handling Exceptions

- In example below, the user can interrupt the program with `Control-C` or the `stop` button in Jupyter Notebook.
- Note that a user-generated interruption is signalled by raising the **KeyboardInterrupt** exception.

<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
for s in ("0.1", "foo", "1000"):
   try:
     x = int(s)
     print(f' x = {x}')
     break
   except ValueError:
     print("Oops!  That was no valid number.  Try again...")
```

<!-- #region slideshow={"slide_type": "slide"} -->
- A try statement may have more than one except clause
- The optional `else` clause must follow all except clauses.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
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
```

```python slideshow={"slide_type": "slide"}
process_file('workfile.txt') # catch exception return by int() call
```

```python slideshow={"slide_type": "slide"}
# Change permission of the file, workfile.txt cannot be read
!chmod u-r workfile.txt
```

```python slideshow={"slide_type": "fragment"}
process_file('workfile.txt') # catch exception return by open() call
```

```python slideshow={"slide_type": "slide"}
# Let's delete the file workfile.txt
!rm -f workfile.txt
```

```python slideshow={"slide_type": "fragment"}
process_file('workfile.txt') # catch another exception return by open() call
```

```python slideshow={"slide_type": "slide"}
# Insert the value -1 at the top of workfile.txt
!echo "-1" > workfile.txt
%cat workfile.txt
```

```python slideshow={"slide_type": "slide"}
process_file('workfile.txt') # catch exception return by assert()
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Raising Exceptions

The raise statement allows the programmer to force a specified exception to occur.

<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
try:
    raise NameError('HiThere')
except:
    print(sys.exc_info()[0])
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Defining Clean-up Actions

- The try statement has an optional clause which is intended to define clean-up actions that must be executed under all circumstances.

- A finally clause is always executed before leaving the try statement
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
try:
     raise KeyboardInterrupt
except:
    print(sys.exc_info()[0])
finally:
     print('Goodbye, world!')
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Wordcount Exercise
- Improve the function `reduce` to read the results of `words` by using the `KeyError` exception to fill in the dictionary.
 
<!-- #endregion -->
