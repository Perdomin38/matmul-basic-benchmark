"""Generate the shared input matrices used by all three implementations.

For every n in SIZES, A_n.bin and B_n.bin contain n*n little-endian float64
values (row-major) drawn uniformly from [-1, 1) with numpy's default_rng(SEED + n).
Because the three languages read the same files, they multiply exactly the same
logical matrices (no cross-language random generator differences)."""
import numpy as np

from common import INPUT_DIR, SEED, SIZES, VALUE_RANGE, input_paths


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        rng = np.random.default_rng(SEED + n)
        A = rng.uniform(*VALUE_RANGE, size=(n, n))
        B = rng.uniform(*VALUE_RANGE, size=(n, n))
        a_path, b_path = input_paths(n)
        A.astype("<f8").tofile(a_path)
        B.astype("<f8").tofile(b_path)
        print(f"n={n}: wrote {a_path.name}, {b_path.name}")


if __name__ == "__main__":
    main()
