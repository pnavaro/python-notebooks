def narcissitic(n):
    """
    Return True if n is a narcissitic number """

    n_d = len(str(n))
    s = 0
    tmp = n
    while tmp > 0:
        d = tmp % 10
        s += d ** n_d
        tmp //= 10
    return s == n

print([n for n in range(1000, 10000) if narcissitic(n)])
