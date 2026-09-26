"""
A.4 The MLWE matrix operation in both formulations
=====================================================================
A matrix is a list of rows; A[i][j] is the ring element in row i,
column j.

`mlwe_naive` multiplies in the coefficient domain, while `mlwe_ntt`
transforms s and e, forms the product in the transform domain through
`star`, and returns the result to the coefficient domain.

`star` implements the operation (star) of Definition 3.5.1. Its
argument Ahat is the matrix already in the transform domain, which
matches FIPS 204, where the entries of A_hat are generated directly
by ExpandA and never exist in the coefficient representation; this is
also the assumption under which Lemma 4.1.9 is stated.
"""

from ring_arithmetic import ring_add
from ntt import ntt_negacyclic, intt_negacyclic, pointwise_mul
from helpers import poly_str


def mlwe_naive(A, s, e, n, q):
    """u = A s + e, with every product computed in the coefficient domain."""
    from ring_arithmetic import ring_mul_naive

    k, l = len(A), len(s)
    u = []
    for i in range(k):
        acc = [0] * n
        for j in range(l):
            acc = ring_add(acc, ring_mul_naive(A[i][j], s[j], n, q), q)
        u.append(ring_add(acc, e[i], q))
    return u


def star(Ahat, shat, n, q):
    """The product A * s in the transform domain, with pointwise multiplication."""
    k, l = len(Ahat), len(shat)
    out = []
    for i in range(k):
        acc = [0] * n
        for j in range(l):
            acc = ring_add(acc, pointwise_mul(Ahat[i][j], shat[j], q), q)
        out.append(acc)
    return out


def mlwe_ntt(Ahat, s, e, z, n, q):
    """u = A s + e, with the products computed in the transform domain.

    Ahat is the matrix already in the transform domain, as produced by ExpandA.
    """
    shat = [ntt_negacyclic(sj, z, n, q) for sj in s]
    ehat = [ntt_negacyclic(ei, z, n, q) for ei in e]
    uhat = star(Ahat, shat, n, q)
    uhat = [ring_add(uhat[i], ehat[i], q) for i in range(len(uhat))]
    return [intt_negacyclic(ui, z, n, q) for ui in uhat]


if __name__ == "__main__":
    n, q, z = 4, 97, 64
    A = [[[15, 23, 44, 10], [81, 7, 52, 31]],
         [[42, 11, 68, 5], [9, 63, 27, 88]]]
    s = [[1, -1 % q, 0, 1], [-1 % q, 0, 1, -1 % q]]
    e = [[0, 1, -1 % q, 0], [1, 0, 0, 1]]

    u = mlwe_naive(A, s, e, n, q)
    print("coefficient domain")
    for i, ui in enumerate(u, 1):
        print(f"  u{i} = {poly_str(ui)}")

    Ahat = [[ntt_negacyclic(A[i][j], z, n, q) for j in range(2)] for i in range(2)]
    shat = [ntt_negacyclic(sj, z, n, q) for sj in s]
    ehat = [ntt_negacyclic(ei, z, n, q) for ei in e]
    uhat = [ring_add(x, ehat[i], q) for i, x in enumerate(star(Ahat, shat, n, q))]

    print("\ntransform domain")
    print(f"  A11_hat = {Ahat[0][0]}  A12_hat = {Ahat[0][1]}")
    print(f"  s1_hat  = {shat[0]}  s2_hat  = {shat[1]}")
    print(f"  e1_hat  = {ehat[0]}")
    for i, x in enumerate(uhat, 1):
        print(f"  u{i}_hat = {x}")

    print("\n  the same values, obtained by transforming the coefficient-domain result")
    for i, ui in enumerate(u, 1):
        print(f"  N(u{i}) = {ntt_negacyclic(ui, z, n, q)}")

    back = mlwe_ntt(Ahat, s, e, z, n, q)
    print("\nafter the inverse transform")
    for i, ui in enumerate(back, 1):
        print(f"  u{i} = {poly_str(ui)}")

    print("\nboth formulations return the same u:", back == u)

    # Expected output:
    # coefficient domain
    #   u1 = 70 + 76x + 70x^2 + 70x^3
    #   u2 = 64 + 68x + 25x^2 + 43x^3
    #
    # transform domain
    #   A11_hat = [45, 61, 78, 70]  A12_hat = [22, 10, 3, 95]
    #   s1_hat  = [84, 15, 15, 84]  s2_hat  = [68, 10, 71, 41]
    #   e1_hat  = [42, 72, 11, 69]
    #   u1_hat = [80, 20, 36, 47]
    #   u2_hat = [35, 40, 29, 55]
    #
    #   the same values, obtained by transforming the coefficient-domain result
    #   N(u1) = [80, 20, 36, 47]
    #   N(u2) = [35, 40, 29, 55]
    #
    # after the inverse transform
    #   u1 = 70 + 76x + 70x^2 + 70x^3
    #   u2 = 64 + 68x + 25x^2 + 43x^3
    #
    # both formulations return the same u: True
