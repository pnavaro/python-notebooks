---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region slideshow={"slide_type": "slide"} -->
# Strings
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
word = "bonjour"
```

```python slideshow={"slide_type": "fragment"}
print(word, len(word))
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Add a `.` to the variable and then press `<TAB>` to get all attached methods available.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
word.capitalize()
```

<!-- #region slideshow={"slide_type": "fragment"} -->
After choosing your method, press `shift+<TAB>` to get interface.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
word.upper()
```

```python slideshow={"slide_type": "slide"}
help(word.replace) # or word.replace? 
```

```python slideshow={"slide_type": "fragment"}
word.replace('o','O',1)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Strings and `print` Function
Strings can be enclosed in single quotes ('...') or double quotes ("...") with the same result. \ can be used to escape quotes:


<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
print('spam eggs')          # single quotes
print('doesn\'t')           # use \' to escape the single quote...
print("doesn't")            # ...or use double quotes instead
print('"Yes," he said.')    #
print("\"Yes,\" he said.")
print('"Isn\'t," she said.')
```

<!-- #region slideshow={"slide_type": "slide"} -->
`print` function translates C special characters
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
s = '\tFirst line.\nSecond line.'  # \n means newline \t inserts tab
print(s)  # with print(), \n produces a new line
print(r'\tFirst line.\nSecond line.')  # note the r before the quote
```

<!-- #region slideshow={"slide_type": "slide"} -->
## String literals with multiple lines
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
print("""\
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
""") 
```

<!-- #region slideshow={"slide_type": "fragment"} -->
\ character removes the initial newline.

Strings can be concatenated (glued together) with the + operator, and repeated with *
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
3 * ("Re" + 2 * 'n' + 'es ')
```

<!-- #region slideshow={"slide_type": "slide"} -->
Two or more string literals next to each other are automatically concatenated.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
text = ('Put several strings within parentheses '
         'to have them joined together.')
text
```

<!-- #region slideshow={"slide_type": "slide"} -->
Strings can be indexed, with the first character having index 0. There is no separate character type; a character is simply a string of size one
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
word = 'Python @ ENSAI'
print(word[0]) # character in position 0
print(word[5]) # character in position 5
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Indices may also be negative numbers, to start counting from the right
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
print(word[-1])  # last character
print(word[-2])  # second-last character
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Slicing Strings
- Omitted first index defaults to zero, 
- Omitted second index defaults to the size of the string being sliced.
- Step can be set with the third index


<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
print(word[:2])  # character from the beginning to position 2 (excluded)
print(word[4:])  # characters from position 4 (included) to the end
print(word[-2:]) # characters from the second-last (included) to the end
print(word[::-1]) # This is the reversed string!
```

```python slideshow={"slide_type": "fragment"}
word[::2]
```

<!-- #region slideshow={"slide_type": "slide"} -->
Python strings cannot be changed — they are immutable.
If you need a different string, you should create a new or use Lists.


<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
import sys
try:
    word[0] = 'J'
except:
    print(sys.exc_info()[0])
```

```python slideshow={"slide_type": "slide"}
## Some string methods
print(word.startswith('P'))
```

```python slideshow={"slide_type": "slide"}
print(*("\n"+w for w in dir(word) if not w.startswith('_')) )
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Exercise

- Ask user to input a string.
- Print out the string length.
- Check if the last character is equal to the first character.
- Check if this string contains only letters.
- Check if this string is lower case.
- Check if this string is a palindrome. A palindrome is a word, phrase, number, or other sequence of characters which reads the same backward as forward.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
# %load solutions/strings/demo.py
s = input(" Input a new word ?")
print(" The string length is " + str(len(s)))
print(" The first character is equal to the last", s[0] == s[-1])
print(" String " + s + " contains only letters : ", s.isalpha())
print(" String " + s + " is lower case :", s.islower())
print(" String " + s + " is a plalindrome :", s == s[::-1])

```

```python

```

```python

```
