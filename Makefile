all: 01-Introduction.ipynb \
	02-Strings.ipynb \
	03-Lists.ipynb \
	04-Control.Flow.Tools.ipynb \
	05-Modules.ipynb \
	06-Input.And.Output.ipynb \
	07-Errors.and.Exceptions.ipynb \
	08-Classes.ipynb \
	09-Iterators.ipynb \
	10-Multiprocessing.ipynb \
	11-Standard.Library.ipynb \
	12-Matplotlib.ipynb \
	13-Numpy.ipynb \
	14-SciPy.ipynb \
	15-Sympy.ipynb \
	16-Fortran.ipynb \
	17-Cython.ipynb \
	18-Numba.ipynb \
	19-LandauDamping.ipynb \
	20-Maxwell.2D.ipynb \
	21-Gray.Scott.Model.ipynb \
	22-Matplotlib.Animation.ipynb 

.SUFFIXES: .py .ipynb

.py.ipynb:
	jupytext --sync $<

01-Introduction.ipynb         : 01-Introduction.py
02-Strings.ipynb              : 02-Strings.py
03-Lists.ipynb                : 03-Lists.py
04-Control.Flow.Tools.ipynb   : 04-Control.Flow.Tools.py
05-Modules.ipynb              : 05-Modules.py
06-Input.And.Output.ipynb     : 06-Input.And.Output.py
07-Errors.and.Exceptions.ipynb: 07-Errors.and.Exceptions.py
08-Classes.ipynb              : 08-Classes.py
09-Iterators.ipynb            : 09-Iterators.py
10-Multiprocessing.ipynb      : 10-Multiprocessing.py
11-Standard.Library.ipynb     : 11-Standard.Library.py
12-Matplotlib.ipynb           : 12-Matplotlib.py
13-Numpy.ipynb                : 13-Numpy.py
14-SciPy.ipynb                : 14-SciPy.py
15-Sympy.ipynb                : 15-Sympy.py
16-Fortran.ipynb              : 16-Fortran.py
17-Cython.ipynb               : 17-Cython.py
18-Numba.ipynb                : 18-Numba.py
19-LandauDamping.ipynb        : 19-LandauDamping.py
20-Maxwell.2D.ipynb           : 20-Maxwell.2D.py
21-Gray.Scott.Model.ipynb     : 21-Gray.Scott.Model.py
22-Matplotlib.Animation.ipynb : 22-Matplotlib.Animation.py
