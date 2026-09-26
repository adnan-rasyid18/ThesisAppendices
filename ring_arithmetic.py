"""
A.2 Ring arithmetic in R_q = Z_q[x] / <x^n + 1>
=====================================================================
A ring element is a list of n coefficients: a[j] is the coefficient
of x^j.

Multiplication proceeds in two steps. `ring_mul_naive` forms the
product of two polynomials with n coefficients, which has 2n - 1
coefficients, and `negacyclic_reduce` then applies the relation
x^n = -1 by subtracting the coefficient of x^(n+t) from that of x^t.
This is the direct method whose cost is analysed in Lemma 4.1.5.
"""

from helpers import poly_str


def ring_add(a, b, q):
    """Sum of two ring elements, coefficientwise in Z_q."""
    return [(x + y) % q for x, y in zip(a, b)]


def ring_sub(a, b, q):
    """Difference of two ring elements, coefficientwise in Z_q."""
    return [(x - y) % q for x, y in zip(a, b)]


def negacyclic_reduce(c, n, q):
    """Reduce a list of 2n-1 coefficients modulo x^n + 1, using x^n = -1."""
    r = [c[t] % q for t in range(n)]
    for t in range(n - 1):
        r[t] = (r[t] - c[t + n]) % q
    return r


def ring_mul_naive(a, b, n, q):
    """Product of two ring elements, computed directly in the coefficient domain."""
    c = [0] * (2 * n - 1)
    for i in range(n):
        for j in range(n):
            c[i + j] = (c[i + j] + a[i] * b[j]) % q
    return negacyclic_reduce(c, n, q)


if __name__ == "__main__":
    # --- quick check against the examples of the thesis ---
    # Section 3.3: (1 + 2x + 3x^2 + 4x^3)(5 + 6x + 7x^2 + 8x^3) in R_97
    print("product in R_97 :",
          poly_str(ring_mul_naive([1, 2, 3, 4], [5, 6, 7, 8], 4, 97)))
    # Section 3.1: (6 + 3x)(2 + x) in R_7, and the sample b = a*s + e with e = x
    prod = ring_mul_naive([6, 3], [2, 1], 2, 7)
    print("product in R_7  :", poly_str(prod))
    print("sample b  :", poly_str(ring_add(prod, [0, 1], 7)))
    # the relation x^n = -1
    print("x^3 * x in R_97 :",
          poly_str(ring_mul_naive([0, 0, 0, 1], [0, 1, 0, 0], 4, 97)))

    # Expected output:
    # product in R_97 : 41 + 61x + 2x^2 + 60x^3
    # product in R_7  : 2 + 5x
    # sample b  : 2 + 6x
    # x^3 * x in R_97 : 96
