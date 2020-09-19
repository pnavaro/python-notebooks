def narcissitic(n):
    """
    Return True if n is a narcissitic number with 3 digits """
    assert len(str(n)) == 3  # check if n contains 3 digits
    s = 0
    tmp = n
    while tmp > 0:
        d = tmp % 10
        s += d ** 3
        tmp //= 10
    if s == n:
        return True
    else:
        return False


print([n for n in range(100, 1000) if narcissitic(n)])
