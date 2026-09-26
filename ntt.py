"""
A.3 The negacyclic transform and its inverse
=====================================================================
The transform N_zeta of Definition 3.3.3 and its inverse. The
transform is built in two layers: `ntt_cyclic` computes the cyclic
transform by the recursive decomposition of Theorem 3.4.6, and
`ntt_negacyclic` obtains the negacyclic one from it by the twisting
relation of Corollary 3.3.4, multiplying the coefficient a[j] by
zeta^j before applying the cyclic transform with omega = zeta^2.

The powers omega^i required by the butterfly operations are generated
by repeated multiplication rather than by exponentiation. Computing
omega^i afresh for each butterfly would cost O(log n) multiplications
and would raise the total to Theta(n log^2 n), whereas a running
power keeps each butterfly at a constant number of operations, which
is the cost assumed in Theorem 4.1.1.
"""

from ring_arithmetic import ring_add
from helpers import poly_str


def ntt_cyclic(a, w, n, q):
    """Cyclic transform of a with root w, by recursive radix-2 decomposition."""
    if n == 1:
        return list(a)
    w2 = w * w % q
    even = ntt_cyclic(a[0::2], w2, n // 2, q)
    odd = ntt_cyclic(a[1::2], w2, n // 2, q)
    out = [0] * n
    wi = 1  # running power: wi = w^i
    for i in range(n // 2):
        t = wi * odd[i] % q
        out[i] = (even[i] + t) % q
        out[i + n // 2] = (even[i] - t) % q
        wi = wi * w % q
    return out


def twist(a, z, n, q):
    """Multiply the coefficient a[j] by z^j."""
    return [a[j] * pow(z, j, q) % q for j in range(n)]


def untwist(a, z, n, q):
    """Multiply the coefficient a[j] by z^(-j), undoing `twist`."""
    zi = pow(z, -1, q)
    return [a[j] * pow(zi, j, q) % q for j in range(n)]


def ntt_negacyclic(a, z, n, q):
    """Negacyclic transform of a with root z of order 2n."""
    return ntt_cyclic(twist(a, z, n, q), z * z % q, n, q)


def intt_negacyclic(ahat, z, n, q):
    """Inverse of `ntt_negacyclic`."""
    wi = pow(z * z % q, -1, q)
    ni = pow(n, -1, q)
    b = ntt_cyclic(ahat, wi, n, q)
    return untwist([ni * x % q for x in b], z, n, q)


def pointwise_mul(a, b, q):
    """Pointwise product of two vectors in the transform domain."""
    return [x * y % q for x, y in zip(a, b)]


if __name__ == "__main__":
    from ring_arithmetic import ring_mul_naive

    n, q, z = 4, 97, 64
    a, b = [1, 2, 3, 4], [5, 6, 7, 8]

    ahat = ntt_negacyclic(a, z, n, q)
    bhat = ntt_negacyclic(b, z, n, q)
    chat = pointwise_mul(ahat, bhat, q)
    c = intt_negacyclic(chat, z, n, q)

    print("a_hat :", ahat)
    print("b_hat :", bhat)
    print("c_hat :", chat)
    print("c  :", poly_str(c))
    print("naive :", poly_str(ring_mul_naive(a, b, n, q)))

    # the transform computed by decomposition agrees with direct evaluation
    s1 = [1, -1 % q, 0, 1]
    points = [pow(z, 2 * i + 1, q) for i in range(n)]
    direct = [sum(s1[j] * pow(r, j, q) for j in range(n)) % q for r in points]
    print("s1_hat by decomposition :", ntt_negacyclic(s1, z, n, q))
    print("s1_hat by evaluation  :", direct)

    # Expected output:
    # a_hat : [7, 0, 30, 64]
    # b_hat : [70, 81, 54, 9]
    # c_hat : [5, 0, 68, 91]
    # c  : 41 + 61x + 2x^2 + 60x^3
    # naive : 41 + 61x + 2x^2 + 60x^3
    # s1_hat by decomposition : [84, 15, 15, 84]
    # s1_hat by evaluation  : [84, 15, 15, 84]
