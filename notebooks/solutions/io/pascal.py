def binomial(n, p):
    b = 1
    for i in range(1, min(p, n - p) + 1):
        b *= n
        b = b // i
        n -= 1
    return b


def pascal_triangle(n):
    for i in range(n):
        line = [binomial(i, j) for j in range(i + 1)]
        s = (n - i) * 3 * " "  # number of spaces
        for c in line:
            s += repr(c).rjust(3) + 3 * " "  # coeffs repr split by 3 spaces
        print(s)


pascal_triangle(10)
