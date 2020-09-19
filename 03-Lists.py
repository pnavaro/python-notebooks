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
# # Python lists and tuples
#
# - List is the most versatile Python data type to group values with others
# - Can be written as a list of comma-separated values (items) between square brackets. 
# - Tuples are written between parenthesis. They are read-only lists.
# - Lists can contain items of different types.
# - Like strings, lists can be indexed and sliced.
# - Lists also support operations like concatenation.
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Indexing

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.744869Z", "iopub.execute_input": "2020-09-12T13:29:12.745779Z", "iopub.status.idle": "2020-09-12T13:29:12.748472Z", "shell.execute_reply": "2020-09-12T13:29:12.749090Z"}}
squares = [1, 4, 9, 16, 25]
print(squares)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.753403Z", "iopub.execute_input": "2020-09-12T13:29:12.754254Z", "iopub.status.idle": "2020-09-12T13:29:12.756788Z", "shell.execute_reply": "2020-09-12T13:29:12.757376Z"}}
print(squares[0])  # indexing returns the item

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.768834Z", "iopub.execute_input": "2020-09-12T13:29:12.769889Z", "iopub.status.idle": "2020-09-12T13:29:12.772598Z", "shell.execute_reply": "2020-09-12T13:29:12.773214Z"}}
squares[-1]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.777553Z", "iopub.execute_input": "2020-09-12T13:29:12.778543Z", "iopub.status.idle": "2020-09-12T13:29:12.781187Z", "shell.execute_reply": "2020-09-12T13:29:12.781793Z"}}
squares[-3:] # slicing returns a new list

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.786015Z", "iopub.execute_input": "2020-09-12T13:29:12.787020Z", "iopub.status.idle": "2020-09-12T13:29:12.789480Z", "shell.execute_reply": "2020-09-12T13:29:12.790070Z"}}
squares += [36, 49, 64, 81, 100]
print(squares)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# - Unlike strings, which are immutable, lists are a mutable type.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.794693Z", "iopub.execute_input": "2020-09-12T13:29:12.795701Z", "iopub.status.idle": "2020-09-12T13:29:12.798023Z", "shell.execute_reply": "2020-09-12T13:29:12.798905Z"}}
cubes = [1, 8, 27, 65, 125]  # something's wrong here  
cubes[3] = 64  # replace the wrong value, the cube of 4 is 64, not 65!
print(cubes)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.803072Z", "iopub.execute_input": "2020-09-12T13:29:12.804054Z", "iopub.status.idle": "2020-09-12T13:29:12.806388Z", "shell.execute_reply": "2020-09-12T13:29:12.806971Z"}}
cubes.append(216)  # add the cube of 6
print(cubes)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.810801Z", "iopub.execute_input": "2020-09-12T13:29:12.811662Z", "shell.execute_reply": "2020-09-12T13:29:12.814722Z", "iopub.status.idle": "2020-09-12T13:29:12.814099Z"}}
cubes.remove(1)
print(cubes)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Assignment 
#
# - You can change the size of the list or clear it entirely.
# - The built-in function len() returns list size.
# - It is possible to create lists containing other lists.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.819266Z", "iopub.execute_input": "2020-09-12T13:29:12.820266Z", "iopub.status.idle": "2020-09-12T13:29:12.822565Z", "shell.execute_reply": "2020-09-12T13:29:12.823166Z"}}
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
letters[2:5] = ['C', 'D', 'E'] # replace some values
print(letters)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.827288Z", "iopub.execute_input": "2020-09-12T13:29:12.828204Z", "iopub.status.idle": "2020-09-12T13:29:12.830560Z", "shell.execute_reply": "2020-09-12T13:29:12.831184Z"}}
letters[2:5] = [] # now remove them
print(letters)

# + {"slideshow": {"slide_type": "slide"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.835444Z", "iopub.execute_input": "2020-09-12T13:29:12.836289Z", "shell.execute_reply": "2020-09-12T13:29:12.838730Z", "iopub.status.idle": "2020-09-12T13:29:12.838009Z"}}
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.842876Z", "iopub.execute_input": "2020-09-12T13:29:12.843836Z", "iopub.status.idle": "2020-09-12T13:29:12.846357Z", "shell.execute_reply": "2020-09-12T13:29:12.847061Z"}}
x

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.851527Z", "iopub.execute_input": "2020-09-12T13:29:12.852505Z", "iopub.status.idle": "2020-09-12T13:29:12.855065Z", "shell.execute_reply": "2020-09-12T13:29:12.855676Z"}}
x[0]

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.860214Z", "iopub.execute_input": "2020-09-12T13:29:12.861181Z", "iopub.status.idle": "2020-09-12T13:29:12.863580Z", "shell.execute_reply": "2020-09-12T13:29:12.864191Z"}}
x[0][1], len(x)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Assignment, Copy and Reference

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.868528Z", "iopub.execute_input": "2020-09-12T13:29:12.869398Z", "iopub.status.idle": "2020-09-12T13:29:12.871666Z", "shell.execute_reply": "2020-09-12T13:29:12.872317Z"}}
a = [0, 1, 2, 3, 4]
b = a
print("b = ",b)

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.876246Z", "iopub.execute_input": "2020-09-12T13:29:12.877167Z", "iopub.status.idle": "2020-09-12T13:29:12.879591Z", "shell.execute_reply": "2020-09-12T13:29:12.880297Z"}}
b[1] = 20        # Change one value in b
print("a = ",a) # Y

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# **b is a reference to a, they occupy same space memory**

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.884903Z", "iopub.execute_input": "2020-09-12T13:29:12.885869Z", "iopub.status.idle": "2020-09-12T13:29:12.888204Z", "shell.execute_reply": "2020-09-12T13:29:12.888832Z"}}
b = a[:] # assign a slice of a and you create a new list
b[2] = 10
print("b = ",b)
print("a = ",a)   

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Some useful List Methods

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.893229Z", "iopub.execute_input": "2020-09-12T13:29:12.894142Z", "iopub.status.idle": "2020-09-12T13:29:12.896989Z", "shell.execute_reply": "2020-09-12T13:29:12.897631Z"}}
a = list("Python-2020")
a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.901969Z", "iopub.execute_input": "2020-09-12T13:29:12.902916Z", "iopub.status.idle": "2020-09-12T13:29:12.905498Z", "shell.execute_reply": "2020-09-12T13:29:12.906111Z"}}
a.sort()
a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.910806Z", "iopub.execute_input": "2020-09-12T13:29:12.911751Z", "iopub.status.idle": "2020-09-12T13:29:12.914179Z", "shell.execute_reply": "2020-09-12T13:29:12.914883Z"}}
a.reverse()
a

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.919337Z", "iopub.execute_input": "2020-09-12T13:29:12.920278Z", "shell.execute_reply": "2020-09-12T13:29:12.923543Z", "iopub.status.idle": "2020-09-12T13:29:12.922958Z"}}
a.pop() #pop the last item and remove it from the list
a

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Dictionary
#
# They are indexed by keys, which are often strings.

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.927923Z", "iopub.execute_input": "2020-09-12T13:29:12.928886Z", "shell.execute_reply": "2020-09-12T13:29:12.931068Z", "iopub.status.idle": "2020-09-12T13:29:12.930464Z"}}
person = dict(firstname="John", lastname="Smith", email="john.doe@domain.fr")
person['size'] = 1.80
person['weight'] = 70

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.935230Z", "iopub.execute_input": "2020-09-12T13:29:12.936239Z", "iopub.status.idle": "2020-09-12T13:29:12.938782Z", "shell.execute_reply": "2020-09-12T13:29:12.939389Z"}}
person

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.943432Z", "iopub.execute_input": "2020-09-12T13:29:12.944387Z", "iopub.status.idle": "2020-09-12T13:29:12.946841Z", "shell.execute_reply": "2020-09-12T13:29:12.947482Z"}}
print(person.keys())

# + {"slideshow": {"slide_type": "fragment"}, "execution": {"iopub.status.busy": "2020-09-12T13:29:12.951453Z", "iopub.execute_input": "2020-09-12T13:29:12.952314Z", "iopub.status.idle": "2020-09-12T13:29:12.954642Z", "shell.execute_reply": "2020-09-12T13:29:12.955291Z"}}
print(person.items())

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Exercises
#
# - Split the string "python ENSAI 2020" into the list ["python","ENSAI", 2020]
# - Insert "september" and value 7 before 2020 in the result list.
# - Capitalize the first item to "Python"
# - Create a dictionary with following keys (meeting, month, day, year)
# - Print out the items.
# - Append the key "place" to this dictionary and set the value to "ENSAI".
# ```python
# ['python', 'ENSAI', '2020']
# ['python', 'ENSAI', 'september', 7, '2020']
# ['Python', 'ENSAI', 'september', 7, '2020']
# {'course': 'Python','september': 'september','day': 7,'year': '2020','place': 'ENSAI'}
#  ```
