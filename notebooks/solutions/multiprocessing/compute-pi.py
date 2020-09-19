import time, random, math
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

def compute_pi(n):
    count = 0
    for i in range(n):
        x=random.random()
        y=random.random()
        if x*x + y*y <= 1: count+=1
    return count

times = []
for np in range(1,cpu_count()+1):
    elapsed_time = time.time()
    n = 4 * 10**7
    part_count=[n//np] * np
    with ProcessPoolExecutor(np) as pool: 
        count=pool.map(compute_pi, part_count)
    pi = 4*sum(count)/n
    print ("Number of cores {0}, Error : {1:.8f}"
       " time : {2:.8f}".format(np, abs(pi-math.pi) ,time.time()-elapsed_time))
    times.append(time.time()-elapsed_time)


# plot the speed-up of your solution ...
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
procs = [p+1 for p in range(len(times))]
etimes = [times[0]/t for t,p in zip(times,procs)]
plt.plot(procs,etimes,'b-o', procs, procs, 'r-');
