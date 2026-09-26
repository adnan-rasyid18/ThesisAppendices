# Source Code — Analysis of the Number-Theoretic Transform (NTT) on Module Learning With Errors (MLWE)

Companion code for the bachelor's thesis *Analysis of the Number-Theoretic
Transform (NTT) on Module Learning With Errors (MLWE)* (Bagas Adnan
Rasyid, Department of Mathematics, Universitas Gadjah Mada, 2026).

This repository is a tidied, split-file version of the thesis's Appendix A
notebook, originally developed in Google Colab. It verifies the worked
examples of Chapter III and reproduces the execution-time measurements of
Chapter IV (Tables 4.4 and 4.5).

## Files

| File | Thesis section | Contents |
|---|---|---|
| `helpers.py` | A.1 | Modular inverse, centered representatives, generators, roots of unity |
| `ring_arithmetic.py` | A.2 | Arithmetic in `R_q = Z_q[x] / <x^n + 1>` |
| `ntt.py` | A.3 | The negacyclic NTT (`ntt_negacyclic`, `intt_negacyclic`) and its cyclic core |
| `mlwe.py` | A.4 | The MLWE matrix operation `u = As + e`, naive and transform-based |
| `verify_examples.py` | A.5 | Exhaustive-search verification of the Chapter III examples (Search-LWE/RLWE/MLWE, Decision-LWE) |
| `measure_environment.py` | A.6 | Records the machine the benchmarks run on (Table 4.3) |
| `benchmark.py` | A.7–A.8 | Execution-time and storage measurements (Tables 4.4 and 4.5) |

Each module's `if __name__ == "__main__":` block reproduces the
corresponding "Output A.n" listing in the thesis, so running a file
directly checks it against the printed expected output in a comment block
at the end.

## Requirements

Python 3.10+ and the standard library only — no third-party packages.
`measure_environment.py` additionally shells out to `lscpu`, `free`, and
(optionally) `nvidia-smi`, which are standard on Linux/Colab but may not
exist on macOS or Windows; it isn't needed to run the other scripts.

## Running

```bash
# Verify a single module against its worked example
python3 helpers.py
python3 ring_arithmetic.py
python3 ntt.py
python3 mlwe.py

# Verify the exhaustive-search examples of Chapter III
python3 verify_examples.py

# Record the current machine's specification (Table 4.3)
python3 measure_environment.py

# Reproduce the timing and storage measurements (Tables 4.4 and 4.5)
python3 benchmark.py
```

`benchmark.py` takes a few minutes to run in full, mostly on the
n = 1024 point of the ring-parameter sweep (A.8). All random data is
generated from the fixed seed `SEED = 2026` defined in `helpers.py`, so
the correctness checks and the reported numbers should reproduce
run-to-run on the same machine; absolute timings will differ across
hardware.

## Notes on this version vs. the thesis appendix

This split-file version behaves identically to the appendix printed in
the thesis, with two small differences:

- The f-string in the original `benchmark.py`-equivalent print statement
  (thesis p. 122) is reflowed here as adjacent string literals — the
  original line-wrap in the PDF is not valid Python if copied verbatim.
- `measure_environment.py` uses English identifiers and messages; the
  logic is unchanged from the thesis listing.

Everything else — every function, constant, and printed value — matches
the thesis appendix exactly.

## Provenance

Original notebook: `https://bit.ly/AdnanThesisAppendices` (see thesis,
Appendix A, p. 103). This repository is the code from that notebook,
reorganized into importable modules; consider tagging a release (and,
optionally, connecting the repo to Zenodo for a permanent DOI) so the
thesis can cite a fixed snapshot instead of the notebook link.
