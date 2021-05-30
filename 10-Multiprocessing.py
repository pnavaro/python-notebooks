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
# # Multiprocessing

# + {"slideshow": {"slide_type": "fragment"}}
from multiprocessing import cpu_count

cpu_count()

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Map reduce example

# + {"slideshow": {"slide_type": "fragment"}}
from time import sleep
def delayed_square(x):
    sleep(1)
    return x*x
data = list(range(8))
data

# + {"slideshow": {"slide_type": "fragment"}}
%time sum(delayed_square(x) for x in data)

# + {"slideshow": {"slide_type": "fragment"}}
%time sum(map(delayed_square,data))

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# We can process each `delayed_square` calls independently and in parallel.  To accomplish this we'll apply that function across all list items in parallel using multiple processes.
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Thread and Process: Differences
#
# - A Process is an instance of a running program. 
# - Process may contain one or more threads, but a thread cannot contain a process.
# - Process has a self-contained execution environment. It has its own memory space. 
# - Application running on your computer may be a set of cooperating processes.
#
# - A Thread is made of and exist within a Process; every process has at least one. 
# - Multiple threads in a process share resources, which helps in efficient communication between threads.
# - Threads can be concurrent on a multi-core system, with every core executing the separate threads simultaneously.
#
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Multi-Processing vs Multi-Threading
#
# ### Memory
# - Each process has its own copy of the data segment of the parent process.
# - Each thread has direct access to the data segment of its process.
# - A process runs in separate memory spaces.
# - A thread runs in shared memory spaces.
#
# ### Communication
# - Processes must use inter-process communication to communicate with sibling processes.
# - Threads can directly communicate with other threads of its process.
#
# ### Overheads
# - Processes have considerable overhead.
# - Threads have almost no overhead.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Multi-Processing vs Multi-Threading
#
# ### Creation
# - New processes require duplication of the parent process.
# - New threads are easily created.  
#
# ### Control
# - Processes can only exercise control over child processes.
# - Threads can exercise considerable control over threads of the same process.
#
# ### Changes
# - Any change in the parent process does not affect child processes.
# - Any change in the main thread may affect the behavior of the other threads of the process.
#

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## The Global Interpreter Lock (GIL)
#
# - The Python interpreter is not thread safe.
# - A few critical internal data structures may only be accessed by one thread at a time. Access to them is protected by the GIL.
# - Attempts at removing the GIL from Python have failed until now. The main difficulty is maintaining the C API for extension modules.
# - Multiprocessing avoids the GIL by having separate processes which each have an independent copy of the interpreter data structures.
# - The price to pay: serialization of tasks, arguments, and results.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Multiprocessing (history)
#
# - The multiprocessing allows the programmer to fully leverage multiple processors. 
# - The `Pool` object parallelizes the execution of a function across multiple input values.
# - The if `__name__ == '__main__'` part is necessary.
# <p><font color=red> The next program does not work in a cell you need to save it and run with python in a terminal </font></p>
#
# ```bash
# python3 pool.py
# ```

# + {"slideshow": {"slide_type": "fragment"}}
%%file pool.py

from time import time, sleep
    
from multiprocessing import Pool

def delayed_square(x):
    sleep(1)
    return x*x

if __name__ == '__main__': # Executed only on main process.
    start = time()
    data = list(range(8))
    with Pool() as p:
        result = sum(p.map(delayed_square, data))
    stop = time()
    print(f"result = {result} - Elapsed time {stop - start}")
# -

import sys
!{sys.executable} pool.py

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Futures
#
# The `concurrent.futures` module provides a high-level interface for asynchronously executing callables.
#
# The asynchronous execution can be performed with threads, using ThreadPoolExecutor, or separate processes, using ProcessPoolExecutor. Both implement the same interface, which is defined by the abstract Executor class.

# + {"slideshow": {"slide_type": "slide"}}
%%file process_pool.py
import os
from time import time, sleep
if os.name == "nt":
    from loky import ProcessPoolExecutor  # for Windows users 
else:
    from concurrent.futures import ProcessPoolExecutor

    from time import time, sleep
    
def delayed_square(x):
    sleep(1)
    return x*x

if __name__ == "__main__":
    start = time()
    data = list(range(8))
    with ProcessPoolExecutor() as pool:
        result = sum(pool.map(delayed_square, data))
    stop = time()
    print(f" result : {result} - elapsed time {stop - start}")

# + {"slideshow": {"slide_type": "slide"}}
!{sys.executable} process_pool.py

# + {"slideshow": {"slide_type": "slide"}}
%%time
from concurrent.futures import ThreadPoolExecutor

e = ThreadPoolExecutor()

results = list(e.map(delayed_square, range(8)))

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Asynchronous Future
# While many parallel applications can be described as maps, some can be more complex. In this section we look at the asynchronous Future interface, which provides a simple API for ad-hoc parallelism. This is useful for when your computations don't fit a regular pattern.
#

# + [markdown] {"slideshow": {"slide_type": "fragment"}}
# ### Executor.submit
#
# The `submit` method starts a computation in a separate thread or process and immediately gives us a `Future` object that refers to the result.  At first, the future is pending.  Once the function completes the future is finished. 
#
# We collect the result of the task with the `.result()` method,
# which does not return until the results are available.

# + {"slideshow": {"slide_type": "slide"}}
from time import sleep

def slowadd(a, b, delay=1):
    sleep(delay)
    return a + b


# + {"slideshow": {"slide_type": "fragment"}}
from concurrent.futures import ThreadPoolExecutor
e = ThreadPoolExecutor(4)
future = e.submit(slowadd, 1, 2)
future

# + {"slideshow": {"slide_type": "fragment"}}
future.result()

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# Submit many tasks all at once and they be will executed in parallel.

# + {"slideshow": {"slide_type": "fragment"}}
%%time
results = [slowadd(i, i, delay=1) for i in range(8)]

# + {"slideshow": {"slide_type": "fragment"}}
%%time
futures = [e.submit(slowadd, 1, 1, delay=1) for i in range(8)]
results = [f.result() for f in futures]

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# *  Submit fires off a single function call in the background, returning a future.  
# *  When you combine submit with a single for loop we recover the functionality of map.  
# *  To collect your results, replace each of futures, `f`, with a call to `f.result()`
# *  Combine submit with multiple for loops and other general programming to get something more general than map.
# *  Sometimes, it did not speed up the code very much
# *  Threads and processes show some performance differences
# *  Use threads carefully, you can break your Python session.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# Today most library designers are coordinating around the concurrent.futures interface, so it's wise to move over.
#
# * Profile your code
# * Used concurrent.futures.ProcessPoolExecutor for simple parallelism 
# * Gained some speed boost (but not as much as expected)
# * Lost ability to diagnose performance within parallel code
# * Describing each task as a function call helps use tools like map for parallelism
# * Making your tasks fast is often at least as important as parallelizing your tasks.

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ### Exercise: Pi computation
#
# Parallelize this computation with a ProcessPoolExecutor. ThreadPoolExecutor is not usable because of `random` function calls.

# + {"slideshow": {"slide_type": "fragment"}}
import time
import random

def compute_pi(n):
    count = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            count += 1
    return count

elapsed_time = time.time()
nb_simulations = 4
n = 10**7
result = [compute_pi(n) for i in range(nb_simulations)]
pi = 4 * sum(result) / (n*nb_simulations)
print(f"Estimated value of Pi : {pi:.8f} time : {time.time()-elapsed_time:.8f}")
# -

# ### Exercise
#
# - Do the same computation using asynchronous future
# - Implement a joblib version (see example below)

# + [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Parallel tools for Python
#
# The parallel tools from standard library are very limited. You will have more powerful features with:
#
# - [Joblib](https://joblib.readthedocs.io/en/latest/) provides a simple helper class to write parallel for loops using multiprocessing.
# - [Dask](https://dask.org)
# - [PySpark](https://spark.apache.org/docs/latest/api/python/index.html)
# - [mpi4py](https://mpi4py.readthedocs.io)
#
