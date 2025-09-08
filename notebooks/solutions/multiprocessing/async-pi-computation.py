import time, random
from concurrent.futures import ProcessPoolExecutor
def compute_pi(n):
    count = 0
    for i in range(n):
        x=random.random()
        y=random.random()
        if x*x + y*y <= 1: count+=1
    return count
    
elapsed_time = time.time()
np = 4
n = 10**7

pool = ProcessPoolExecutor()

futures = [pool.submit(compute_pi,n)] * np

results = [f.result() for f in futures]
    
pi = 4* sum(results)/ (n*np)
print ("Estimated value of Pi : {0:.8f} time : {1:.8f}".format(pi,time.time()-elapsed_time))
