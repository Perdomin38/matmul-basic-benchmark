"""Correctness checks run BEFORE benchmarking.

Each implementation multiplies small test matrices and writes its result to a
file, which is compared with a known answer or with NumPy's A @ B (NumPy is
only the trusted reference, never one of the measured implementations).

Tests
  known_2x2  : [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]   (exact)
  identity   : A x I = A, n = 37                                   (exact)
  zero       : A x 0 = 0, n = 37                                   (exact)
  random_100 : random A, B with n = 100 vs NumPy                   (tolerance)

Tolerance for the random test: |C - C_ref| <= ATOL + RTOL * |C_ref| with
RTOL = ATOL = 1e-9. Rounding errors of a length-n float64 dot product are
roughly n * 1.1e-16 * sum|a_k b_k| ~ 1e-13 here, so the tolerance is far above
legitimate rounding differences (NumPy/BLAS sums in a different order) and far
below the O(1) errors that an indexing bug would produce.
Results are written to results/raw/correctness.csv."""
import csv
import subprocess
import sys

import numpy as np

from common import LANGUAGES, RAW_DIR, TEST_DIR, command

RTOL = 1e-9
ATOL = 1e-9


def test_cases():
    rng = np.random.default_rng(12345)
    A37 = rng.uniform(-1, 1, (37, 37))
    A100 = rng.uniform(-1, 1, (100, 100))
    B100 = rng.uniform(-1, 1, (100, 100))
    return [
        ("known_2x2", np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]),
         np.array([[19.0, 22.0], [43.0, 50.0]]), True),
        ("identity", A37, np.eye(37), A37, True),
        ("zero", A37, np.zeros((37, 37)), np.zeros((37, 37)), True),
        ("random_100", A100, B100, A100 @ B100, False),
    ]


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    all_ok = True
    for name, A, B, expected, exact in test_cases():
        n = A.shape[0]
        a_path, b_path = TEST_DIR / f"{name}_A.bin", TEST_DIR / f"{name}_B.bin"
        A.astype("<f8").tofile(a_path)
        B.astype("<f8").tofile(b_path)
        results = {}
        for lang in LANGUAGES:
            out_path = TEST_DIR / f"{name}_C_{lang}.bin"
            subprocess.run(command(lang, n, a_path, b_path, 0, 1, out_path),
                           check=True, capture_output=True)
            C = np.fromfile(out_path, dtype="<f8").reshape(n, n)
            results[lang] = C
            max_err = float(np.max(np.abs(C - expected)))
            ok = bool(np.array_equal(C, expected)) if exact else \
                bool(np.allclose(C, expected, rtol=RTOL, atol=ATOL))
            same_as_c = bool(np.array_equal(C, results["c"]))
            all_ok &= ok
            rows.append({"test": name, "n": n, "language": lang,
                         "criterion": "exact" if exact else f"rtol={RTOL},atol={ATOL}",
                         "max_abs_error": f"{max_err:.3e}", "passed": ok,
                         "bitwise_equal_to_c": same_as_c})
            print(f"{name:11s} n={n:3d} {lang:6s} max_abs_err={max_err:.3e} "
                  f"passed={ok} bitwise_equal_to_c={same_as_c}")

    with open(RAW_DIR / "correctness.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
