"""
A.1 Helper functions
=====================================================================
Inversion in Z_q, the centered representative of a residue class, and
the search for generators and roots of unity.

`primitive_root_of_unity` implements the construction in the proof of
Theorem 3.2.4: it finds a generator g of Z_q^* by testing
g^((q-1)/p) != 1 for every prime p dividing q - 1, and returns
g^((q-1)/order). It returns None when the requested order does not
divide q - 1, which is exactly when no such root exists.
"""

SEED = 2026


def modinv(a, q):
    """Multiplicative inverse of a in Z_q."""
    return pow(a, -1, q)


def centered(a, q):
    """Representative of a in {-(q-1)/2, ..., (q-1)/2}."""
    a %= q
    return a - q if a > q // 2 else a


def prime_factors(m):
    """Set of distinct prime factors of m."""
    factors, d = set(), 2
    while d * d <= m:
        while m % d == 0:
            factors.add(d)
            m //= d
        d += 1
    if m > 1:
        factors.add(m)
    return factors


def is_generator(g, q):
    """True if g generates the multiplicative group Z_q^*."""
    return all(pow(g, (q - 1) // p, q) != 1 for p in prime_factors(q - 1))


def find_generator(q):
    """Smallest generator of Z_q^*."""
    g = 2
    while not is_generator(g, q):
        g += 1
    return g


def element_order(a, q):
    """Multiplicative order of a in Z_q^*."""
    k, x = 1, a % q
    while x != 1:
        x = x * a % q
        k += 1
    return k


def primitive_root_of_unity(order, q):
    """Primitive root of unity of the given order in Z_q, or None if none exists."""
    if (q - 1) % order != 0:
        return None
    return pow(find_generator(q), (q - 1) // order, q)


def is_primitive_root_of_unity(a, order, q):
    """True if a has multiplicative order exactly `order` in Z_q^*."""
    return element_order(a, q) == order


def poly_str(coeffs, var="x"):
    """Readable form of a coefficient list, e.g. [1, 2, 0, 3] -> '1 + 2x + 3x^3'."""
    terms = []
    for j, c in enumerate(coeffs):
        if c == 0:
            continue
        if j == 0:
            terms.append(f"{c}")
        elif c == 1:
            terms.append(var if j == 1 else f"{var}^{j}")
        else:
            terms.append(f"{c}{var}" if j == 1 else f"{c}{var}^{j}")
    return " + ".join(terms) if terms else "0"


if __name__ == "__main__":
    # --- quick check against the examples of the thesis ---
    print("generator of Z_7^*  :", find_generator(7))
    print("generator of Z_97^* :", find_generator(97))
    print("omega (n = 4, q = 97)  :", primitive_root_of_unity(4, 97))
    print("zeta  (2n = 8, q = 97) :", primitive_root_of_unity(8, 97))
    # the root specified in FIPS 204 for the ML-DSA parameters
    print("zeta (ML-DSA parameters) :", 1753)
    print("ord(1753) in Z_8380417 :", element_order(1753, 8380417))

    # Expected output:
    # generator of Z_7^*  : 3
    # generator of Z_97^* : 5
    # omega (n = 4, q = 97)  : 22
    # zeta  (2n = 8, q = 97) : 64
    # zeta (ML-DSA parameters) : 1753
    # ord(1753) in Z_8380417 : 512
