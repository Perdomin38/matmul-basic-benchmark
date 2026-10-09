"""
Basic dense matrix multiplication C = A x B (n x n, float64) in pure Python.

Storage: a list of n row lists. Each element is a Python float object
(an IEEE-754 float64 boxed in a heap object); the lists store pointers
to those objects. No NumPy is used for the computation.
Algorithm: textbook triple loop in i-j-k order, O(n^3) multiply-adds.

Usage: python matmul.py <n> <A.bin> <B.bin> <warmups> <reps> [C_out.bin]
  Input files: n*n little-endian float64 values, row-major.
  Output (stdout): CSV lines "phase,run,time_s,checksum"
  If C_out.bin is given, the last result matrix is written there.

Timed region: only the triple loop. Reading the inputs and allocating
the result happen before the timer starts.
"""
import struct
import sys
import time


def read_matrix(path, n):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != 8 * n * n:
        raise ValueError(f"file {path} does not contain {n * n} values")
    flat = struct.unpack(f"<{n * n}d", data)
    return [list(flat[i * n:(i + 1) * n]) for i in range(n)]


def write_matrix(path, m, n):
    flat = [value for row in m for value in row]
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{n * n}d", *flat))


def multiply(A, B, C, n):
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s


def main():
    if len(sys.argv) < 6:
        sys.exit("usage: python matmul.py n A.bin B.bin warmups reps [C_out.bin]")
    n = int(sys.argv[1])
    warmups = int(sys.argv[4])
    reps = int(sys.argv[5])

    A = read_matrix(sys.argv[2], n)
    B = read_matrix(sys.argv[3], n)
    C = [[0.0] * n for _ in range(n)]

    print("phase,run,time_s,checksum")
    for r in range(warmups + reps):
        start = time.perf_counter()
        multiply(A, B, C, n)
        elapsed = time.perf_counter() - start

        # The checksum consumes the result, so the work cannot be skipped.
        checksum = 0.0
        for row in C:
            for value in row:
                checksum += value

        is_warmup = r < warmups
        phase = "warmup" if is_warmup else "measured"
        run = r + 1 if is_warmup else r - warmups + 1
        print(f"{phase},{run},{elapsed:.9f},{checksum!r}", flush=True)

    if len(sys.argv) >= 7:
        write_matrix(sys.argv[6], C, n)


if __name__ == "__main__":
    main()
