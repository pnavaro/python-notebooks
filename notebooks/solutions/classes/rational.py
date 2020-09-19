class Rational:
    """ Class representing a rational number"""

    def __init__(self, n, d):
        assert isinstance(n, int)
        assert isinstance(d, int)

        def gcd(x, y):
            if x == 0:
                return y
            elif x < 0:
                return gcd(-x, y)
            elif y < 0:
                return -gcd(x, -y)
            else:
                return gcd(y % x, x)

        g = gcd(n, d)
        self.numer = n // g
        self.denom = d // g

    def __add__(self, other):
        return Rational(self.numer * other.denom + other.numer * self.denom,
                        self.denom * other.denom)

    def __sub__(self, other):
        return Rational(self.numer * other.denom - other.numer * self.denom,
                        self.denom * other.denom)

    def __mul__(self, other):
        return Rational(self.numer * other.numer, self.denom * other.denom)

    def __truediv__(self, other):
        return Rational(self.numer * other.denom, self.denom * other.numer)

    def __repr__(self):
        return f"{self.numer:d} / {self.denom:d}"


if __name__ == "__main__":
    r1 = Rational(2, 3)
    r2 = Rational(3, 4)
    print(f"r1 = {r1}")
    print(f"r2 = {r2}")
    print(f"r1-r2 = {r1-r2}")
    print(f"r1+r2 = {r1+r2}")
    print(f"r1*r2 = {r1*r2}")
    print(f"r1/r2 = {r1/r2}")
