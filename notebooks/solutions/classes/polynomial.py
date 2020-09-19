class Polynomial:
    """ Polynomial """

    def __init__(self, coefficients):
        self.coeffs = coefficients
        self.degree = len(coefficients)

    def diff(self, n):
        """ Return the nth derivative """
        coeffs = self.coeffs[:]
        for k in range(n):
            coeffs = [i * coeffs[i] for i in range(1, len(coeffs))]
        return Polynomial(coeffs)

    def __repr__(self):
        output = ""
        for i, c in enumerate(self.coeffs):
            if c != 0:
                if i == 0:
                    output += " {0:+d} ".format(c)
                elif i == 1:
                    output += " {0:+d}x ".format(c)
                else:
                    output += " {0:+d}x^{1} ".format(c, i)

        return output

    def __eq__(self, other):  # override '=='
        return self.coeffs == other.coeffs

    def __add__(self, other):  # ( P + Q )
        if self.degree < other.degree:
            coeffs = self.coeffs + [0] * (other.degree - self.degree)
            return Polynomial([c + q for c, q in zip(other.coeffs, coeffs)])
        else:
            coeffs = other.coeffs + [0] * (self.degree - other.degree)
            return Polynomial([c + q for c, q in zip(self.coeffs, coeffs)])

    def __neg__(self):
        return Polynomial([-c for c in self.coeffs])

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):  # (P * Q) or (alpha * P)

        if isinstance(other, Polynomial):
            _s = self.coeffs
            _q = other.coeffs
            res = [0] * (len(_s) + len(_q) - 1)
            for s_p, s_c in enumerate(_s):
                for q_p, q_c in enumerate(_q):
                    res[s_p + q_p] += s_c * q_c
            return Polynomial(res)
        else:
            return Polynomial([c * other for c in self.coeffs])


if __name__ == "__main__":
    P = Polynomial([-3, -1, 1, -1, 4])
    Q = P.diff(2)
    S = -P
    print(P)
    print(Q)
    print(S)
    print(P + Q)
    print(Q + P)
