#cython: profile=False

def exp_c(double x, int terms = 50):
   cdef double sum
   cdef double power
   cdef double fact
   cdef int i
   sum = 0.
   power = 1.
   fact = 1.
   for i in range(terms):
      sum += power/fact
      power *= x
      fact *= i+1
   return sum
