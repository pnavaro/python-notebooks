n = 100000
k = 0
while n != 1:
    if n & 1:  # returns the last bit of n binary representation.
        n = 3 * n + 1
    else:
        n = n // 2  # Pure division by 2
    k += 1

print(k)
