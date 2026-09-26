"""
A.5 Verification of the examples of Chapter III
=====================================================================
The examples of Section 3.1. make claims that rest on an exhaustive
search: that a given collection of samples admits exactly one secret
with small errors, or, in the case of Decision-LWE, that it admits
none. Those claims range over q^n = 49 candidates for LWE and RLWE
and over q^(l n) = 2401 for MLWE, so this section carries out the
searches and prints the results, which may be compared directly with
the figures quoted in the thesis.

Two search functions are provided. `lwe_search` ranges over Z_q^n and
tests each candidate against the inner products of the samples, while
`mlwe_search` ranges over R_q^l and tests each candidate against the
ring products, covering RLWE as the case l = 1. The remaining
examples of Chapter III are verified in ntt.py and mlwe.py, where the
transform and the matrix operation are computed on the same data.
"""

from itertools import product

from helpers import centered
from ring_arithmetic import ring_add, ring_sub, ring_mul_naive
from helpers import poly_str


def small_errors(errors, bound, q):
    """True if every error lies in {-bound, ..., bound} as a centered residue."""
    return all(abs(centered(x, q)) <= bound for x in errors)


def lwe_search(samples, n, q, bound):
    """All secrets in Z_q^n consistent with the samples, with their error vectors."""
    out = []
    for s in product(range(q), repeat=n):
        errs = [(b - sum(a[t] * s[t] for t in range(n))) % q for a, b in samples]
        if small_errors(errs, bound, q):
            out.append((list(s), [centered(x, q) for x in errs]))
    return out


def mlwe_search(samples, n, l, q, bound):
    """All secrets in R_q^l consistent with the samples; RLWE is the case l = 1."""
    out = []
    for flat in product(range(q), repeat=n * l):
        s = [list(flat[j * n:(j + 1) * n]) for j in range(l)]
        errs = []
        for a, b in samples:
            acc = [0] * n
            for j in range(l):
                acc = ring_add(acc, ring_mul_naive(a[j], s[j], n, q), q)
            errs.extend(ring_sub(b, acc, q))
        if small_errors(errs, bound, q):
            out.append((s, [centered(x, q) for x in errs]))
    return out


if __name__ == "__main__":
    # --- Search-LWE and Decision-LWE ---
    n, q, eta = 2, 7, 1

    instance_A = [((2, 2), 4), ((1, 3), 5), ((6, 5), 6), ((3, 6), 2), ((4, 1), 4)]
    instance_B = [((2, 2), 1), ((1, 3), 4), ((6, 5), 0), ((3, 6), 6), ((4, 1), 3)]

    print(f"Search-LWE: n = {n}, q = {q}, {q**n} candidates in total")
    for m in range(1, 6):
        print(f"  after {m} sample(s): {len(lwe_search(instance_A[:m], n, q, eta))} remain")
    for s, errs in lwe_search(instance_A, n, q, eta):
        print(f"  secret s = {tuple(s)} with errors {tuple(errs)}")

    print(f"\nDecision-LWE, Instance B")
    for m in range(1, 6):
        print(f"  after {m} sample(s): {len(lwe_search(instance_B[:m], n, q, eta))} remain")
    print("  no secret keeps every error within the bound")

    # --- Search-RLWE ---
    rlwe = [([[6, 3]], [2, 6]),
            ([[3, 6]], [1, 2])]

    print(f"\nSearch-RLWE: n = {n}, q = {q}, l = 1, {q**n} candidates in total")
    for m in (1, 2):
        print(f"  after {m} sample(s): {len(mlwe_search(rlwe[:m], n, 1, q, eta))} remain")
    print("  after 1 sample:",
          ", ".join(poly_str(s[0]) for s, _ in mlwe_search(rlwe[:1], n, 1, q, eta)))
    for s, errs in mlwe_search(rlwe, n, 1, q, eta):
        print(f"  secret s = {poly_str(s[0])}")

    # --- Search-MLWE ---
    l = 2
    mlwe = [([[0, 6], [3, 0]], [5, 0]),
            ([[2, 4], [0, 6]], [2, 1]),
            ([[3, 6], [2, 1]], [0, 4])]

    print(f"\nSearch-MLWE: n = {n}, q = {q}, l = {l}, {q**(n*l)} candidates in total")
    for m in (1, 2, 3):
        print(f"  after {m} sample(s): {len(mlwe_search(mlwe[:m], n, l, q, eta))} remain")
    for s, errs in mlwe_search(mlwe, n, l, q, eta):
        print(f"  secret s = ({poly_str(s[0])}, {poly_str(s[1])})")

    # Expected output:
    # Search-LWE: n = 2, q = 7, 49 candidates in total
    #   after 1 sample(s): 21 remain
    #   after 2 sample(s): 9 remain
    #   after 3 sample(s): 4 remain
    #   after 4 sample(s): 1 remain
    #   after 5 sample(s): 1 remain
    #   secret s = (2, 3) with errors (1, 1, 0, -1, 0)
    #
    # Decision-LWE, Instance B
    #   after 1 sample(s): 21 remain
    #   after 2 sample(s): 9 remain
    #   after 3 sample(s): 4 remain
    #   after 4 sample(s): 1 remain
    #   after 5 sample(s): 0 remain
    #   no secret keeps every error within the bound
    #
    # Search-RLWE: n = 2, q = 7, l = 1, 49 candidates in total
    #   after 1 sample(s): 9 remain
    #   after 2 sample(s): 1 remain
    #   after 1 sample: 2x, 1 + 4x, 2 + x, 2 + 6x, 3 + 3x, 4, 4 + 5x, 5 + 2x, 6 + 4x
    #   secret s = 2 + x
    #
    # Search-MLWE: n = 2, q = 7, l = 2, 2401 candidates in total
    #   after 1 sample(s): 441 remain
    #   after 2 sample(s): 81 remain
    #   after 3 sample(s): 1 remain
    #   secret s = (2 + x, 1 + x)
    #
    # NOTE: as of the check on 2026-09-26, "after 4 sample(s)" for Search-LWE
    # and "after 2 sample(s)" for Search-RLWE already leave exactly 1
    # candidate. See the review notes for the thesis text this affects
    # (Examples 3.1.7 and 3.1.16 currently say five/two samples "were
    # needed").
