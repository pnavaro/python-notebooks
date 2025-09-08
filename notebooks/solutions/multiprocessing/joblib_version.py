from joblib import Parallel, delayed

np = 4
n  = 10**7
part_count = [n] * np
result = Parallel(n_jobs=np)(delayed(compute_pi)(i) for i in part_count)

pi = 4* sum(result)/ n

print ("Estimated value of Pi : {0:.8f}".format(pi))
