def exponential(x, terms=50):
    exp = 0.
    power = 1.
    fact = 1.
    for i in range(terms):
        exp += power / fact
        power *= x
        fact *= i + 1
    return exp


print(exponential(1))
