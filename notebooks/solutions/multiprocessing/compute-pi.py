import time, random, math
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

def compute_pi(n):
    count = 0
    for i in range(n):
        x=random.random()
        y=random.random()
        if x*x + y*y <= 1:
            count += 1
    return count

times = []
for nproc in (1, 2, 4, 8, 16):
    elapsed_time = time.time()
    samples = 100000000
    part_count = [samples // nproc] * nproc
    if __name__ == '__main__':
        with ProcessPoolExecutor(nproc) as pool:
            count=pool.map(compute_pi, part_count)

        pi = 4*sum(count)/samples
        times.append(time.time()-elapsed_time)
        print ("Number of cores {0}, Error : {1:.8f}"
           " time : {2:.8f}".format(nproc, abs(pi-math.pi) ,time.time()-elapsed_time))


# plot the speed-up of your solution ...
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
procs = [p+1 for p in range(len(times))]
etimes = [times[0]/t for t,p in zip(times,procs)]
plt.plot(procs,etimes,'b-o', procs, procs, 'r-')

