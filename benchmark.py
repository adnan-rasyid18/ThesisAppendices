"""
A.7 Execution-time measurements at the FIPS 204 parameter sets, and
A.8 Growth of the execution time with the ring dimension
=====================================================================
This script produces the measurements of Table 4.4 and Table 4.5.

For each parameter set (A.7), one instance is generated from the
fixed seed and reused, so that both formulations operate on identical
data. The two results are compared before timing begins, and the run
stops if they differ, so that no timing is reported for an
implementation that does not compute the correct value. Five warm-up
iterations are discarded and the following thirty are recorded, with
the two formulations alternating in order.

The section closes with the comparison of storage discussed in
Subsection 4.2.5.. The number of coefficients held by A and by A_hat
is the same, since N_zeta maps each ring element to a vector of the
same length. The peak temporary allocation is also reported; the
larger figure for the transform-based formulation comes from the
recursion in `ntt_cyclic`, which builds new lists at each level, and
not from the method itself.

A.8 varies the ring dimension while the matrix dimensions are held at
k = l = 4, which are those of ML-DSA-44, so that the point n = 256
measures the same operation as the first row of Table 4.4. The
modulus is kept at q = 8 380 417 throughout, which is possible because
q - 1 = 2^13 * 3 * 11 * 31 admits a primitive root of unity of order
2n for every n = 2^r up to 4096. The last two columns divide the
measured time by the growth term predicted for it by Lemmas 4.1.7 and
4.1.9.
"""

import math
import random
import statistics
import sys
import time
import tracemalloc

from helpers import SEED, primitive_root_of_unity, element_order
from ntt import ntt_negacyclic
from mlwe import mlwe_naive, mlwe_ntt

WARMUP = 5
REPEATS = 30

# n, q and zeta as specified in FIPS 204
N, Q, Z = 256, 8380417, 1753

PARAMETER_SETS = [
    ("ML-DSA-44", 4, 4, 2),
    ("ML-DSA-65", 6, 5, 4),
    ("ML-DSA-87", 8, 7, 2),
]


def generate_instance(k, l, eta, n, q, z, seed):
    """One MLWE instance, with the matrix already transformed."""
    random.seed(seed)
    A = [[[random.randrange(q) for _ in range(n)] for _ in range(l)] for _ in range(k)]
    s = [[random.choice(range(-eta, eta + 1)) % q for _ in range(n)] for _ in range(l)]
    e = [[random.choice(range(-eta, eta + 1)) % q for _ in range(n)] for _ in range(k)]
    Ahat = [[ntt_negacyclic(A[i][j], z, n, q) for j in range(l)] for i in range(k)]
    return A, Ahat, s, e


def summarise(times):
    return {"mean": statistics.mean(times),
            "median": statistics.median(times),
            "sd": statistics.stdev(times)}


def benchmark(name, k, l, eta, n, q, z, seed):
    """Time both formulations on one instance, after verifying that they agree."""
    A, Ahat, s, e = generate_instance(k, l, eta, n, q, z, seed)

    assert mlwe_naive(A, s, e, n, q) == mlwe_ntt(Ahat, s, e, z, n, q), \
        f"{name}: the two formulations disagree"

    for _ in range(WARMUP):
        mlwe_naive(A, s, e, n, q)
        mlwe_ntt(Ahat, s, e, z, n, q)

    t_naive, t_ntt = [], []
    for r in range(REPEATS):
        order = (mlwe_naive, mlwe_ntt) if r % 2 == 0 else (mlwe_ntt, mlwe_naive)
        for f in order:
            t0 = time.perf_counter()
            f(A, s, e, n, q) if f is mlwe_naive else f(Ahat, s, e, z, n, q)
            dt = (time.perf_counter() - t0) * 1000
            (t_naive if f is mlwe_naive else t_ntt).append(dt)

    return summarise(t_naive), summarise(t_ntt)


def total_size(x):
    """Total size in bytes of a nested list of integers."""
    if isinstance(x, list):
        return sum(total_size(y) for y in x) + sys.getsizeof(x)
    return sys.getsizeof(x)


def run_fips204_measurements():
    """A.7: the measurements reported in Table 4.4, plus storage figures."""
    print(f"n = {N}, q = {Q}, zeta = {Z}, order of zeta = {element_order(Z, Q)}")
    print(f"seed = {SEED}, warm-up = {WARMUP} iterations, measured = {REPEATS} iterations")
    print("all times in milliseconds\n")

    header = (f"{'parameter set':<13}{'(k,l)':>8}{'naive mean':>12}{'median':>9}{'sd':>7}"
              f"{'NTT mean':>11}{'median':>9}{'sd':>7}{'S':>7}")
    print(header)
    print("-" * len(header))

    for name, k, l, eta in PARAMETER_SETS:
        naive, ntt = benchmark(name, k, l, eta, N, Q, Z, SEED)
        print(f"{name:<13}{f'({k},{l})':>8}"
              f"{naive['mean']:>12.2f}{naive['median']:>9.2f}{naive['sd']:>7.2f}"
              f"{ntt['mean']:>11.2f}{ntt['median']:>9.2f}{ntt['sd']:>7.2f}"
              f"{naive['median'] / ntt['median']:>7.1f}")

    # --- storage and temporary allocation ---
    name, k, l, eta = PARAMETER_SETS[0]
    A, Ahat, s, e = generate_instance(k, l, eta, N, Q, Z, SEED)

    print(f"\nstorage at {name}")
    print(f"  A  : {total_size(A) / 1024:7.1f} KiB "
          f"({k * l} ring elements, {k * l * N} coefficients)")
    print(f"  A_hat : {total_size(Ahat) / 1024:7.1f} KiB "
          f"({k * l} ring elements, {k * l * N} coefficients)")

    for label, call in (("naive", lambda: mlwe_naive(A, s, e, N, Q)),
                        ("NTT  ", lambda: mlwe_ntt(Ahat, s, e, Z, N, Q))):
        tracemalloc.start()
        call()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  peak temporary allocation, {label} : {peak / 1024:7.1f} KiB")


def run_ring_parameter_sweep():
    """A.8: the measurements reported in Table 4.5."""
    RING_SIZES = [16, 32, 64, 128, 256, 512, 1024]
    SWEEP_WARMUP = 3
    SWEEP_REPEATS = 10
    k_sweep, l_sweep, eta_sweep = 4, 4, 2

    def time_once(f, args):
        t0 = time.perf_counter()
        f(*args)
        return (time.perf_counter() - t0) * 1000

    print(f"\nq = {Q} fixed, k = l = {k_sweep}, eta = {eta_sweep}")
    print(f"seed = {SEED}, warm-up = {SWEEP_WARMUP}, measured = {SWEEP_REPEATS} iterations")
    print("times in milliseconds\n")

    header = (f"{'n':>6}{'zeta':>10}{'naive':>11}{'NTT':>10}{'S':>8}"
              f"{'naive/(kln^2)':>16}{'NTT/(full bound)':>19}")
    print(header)
    print("-" * len(header))

    for n in RING_SIZES:
        z = primitive_root_of_unity(2 * n, Q)
        A, Ahat, s, e = generate_instance(k_sweep, l_sweep, eta_sweep, n, Q, z, SEED)

        assert mlwe_naive(A, s, e, n, Q) == mlwe_ntt(Ahat, s, e, z, n, Q), \
            f"n = {n}: the two formulations disagree"

        for _ in range(SWEEP_WARMUP):
            mlwe_naive(A, s, e, n, Q)
            mlwe_ntt(Ahat, s, e, z, n, Q)

        t_naive, t_ntt = [], []
        for _ in range(SWEEP_REPEATS):
            t_naive.append(time_once(mlwe_naive, (A, s, e, n, Q)))
            t_ntt.append(time_once(mlwe_ntt, (Ahat, s, e, z, n, Q)))

        a = statistics.median(t_naive)
        b = statistics.median(t_ntt)
        predicted_naive = k_sweep * l_sweep * n ** 2
        predicted_ntt = (k_sweep + l_sweep) * n * math.log2(n) + k_sweep * l_sweep * n
        print(f"{n:>6}{z:>10}{a:>11.2f}{b:>10.3f}{a / b:>8.1f}"
              f"{a / predicted_naive * 1000:>16.4f}{b / predicted_ntt * 1000:>19.4f}")


if __name__ == "__main__":
    run_fips204_measurements()
    run_ring_parameter_sweep()

    # Expected output (A.7), times will vary by machine:
    # n = 256, q = 8380417, zeta = 1753, order of zeta = 512
    # seed = 2026, warm-up = 5 iterations, measured = 30 iterations
    # all times in milliseconds
    #
    # parameter set  (k,l)  naive mean  median  sd  NTT mean  median  sd  S
    # -----------------------------------------------------------------------------------
    # ML-DSA-44  (4,4)  216.77  199.95  46.78  12.98  12.29  2.41  16.3
    # ML-DSA-65  (6,5)  433.98  388.25 114.15  20.00  17.40  5.10  22.3
    # ML-DSA-87  (8,7)  808.05  714.89 202.14  28.22  24.84  6.87  28.8
    #
    # storage at ML-DSA-44
    #   A  :  146.8 KiB  (16 ring elements, 4096 coefficients)
    #   A_hat :  145.3 KiB  (16 ring elements, 4096 coefficients)
    #   peak temporary allocation, naive :  71.3 KiB
    #   peak temporary allocation, NTT  :  183.6 KiB
    #
    # Expected output (A.8), times will vary by machine:
    #      n      zeta      naive       NTT       S   naive/(kln^2)   NTT/(full bound)
    # --------------------------------------------------------------------------------
    #     16   6250525       0.83     0.552     1.5          0.2036             0.7186
    #     32   7044481       2.92     1.156     2.5          0.1781             0.6448
    #     64   3241972      11.78     2.573     4.6          0.1798             0.6282
    #    128   6644104      84.77    10.244     8.3          0.3234             1.1116
    #    256   1921994     200.01    12.584    15.9          0.1907             0.6145
    #    512    550930     982.67    26.433    37.2          0.2343             0.5867
    #   1024   1028169    4043.30    57.640    70.1          0.2410             0.5863
