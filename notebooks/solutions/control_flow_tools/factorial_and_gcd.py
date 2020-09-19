def factorial(n):
    """ Return n! """
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def gcd(x, y):
    if x == 0:
        return y
    else:
        return gcd(y % x, x)


print(factorial(5), gcd(12, 16))
