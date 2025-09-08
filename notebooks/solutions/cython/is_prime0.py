
def is_prime0(n):
    if n < 4: return True
    if n % 2 == 0 : return False
    k = 3
    while k*k <= n:
        if n % k == 0: return False
        k += 2
    return True