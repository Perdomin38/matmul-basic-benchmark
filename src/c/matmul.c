/*
 * Basic dense matrix multiplication C = A x B (n x n, float64) in C.
 *
 * Storage: each matrix is ONE contiguous block of n*n doubles in row-major
 * order, so element (i, j) is at index i*n + j.
 * Algorithm: textbook triple loop in i-j-k order, O(n^3) multiply-adds.
 *
 * Usage: matmul <n> <A.bin> <B.bin> <warmups> <reps> [C_out.bin]
 *   Input files: n*n little-endian float64 values, row-major.
 *   Output (stdout): CSV lines "phase,run,time_s,checksum"
 *   If C_out.bin is given, the last result matrix is written there.
 *
 * Timed region: only the triple loop. Reading the inputs and allocating
 * the result happen before the timer starts.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double *read_matrix(const char *path, int n) {
    size_t count = (size_t)n * n;
    double *m = malloc(count * sizeof(double));
    FILE *f = fopen(path, "rb");
    if (m == NULL || f == NULL) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    if (fread(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "file %s does not contain %zu values\n", path, count);
        exit(1);
    }
    fclose(f);
    return m;
}

static void multiply(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s n A.bin B.bin warmups reps [C_out.bin]\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int warmups = atoi(argv[4]);
    int reps = atoi(argv[5]);

    double *A = read_matrix(argv[2], n);
    double *B = read_matrix(argv[3], n);
    double *C = malloc((size_t)n * n * sizeof(double));

    printf("phase,run,time_s,checksum\n");
    for (int r = 0; r < warmups + reps; r++) {
        double start = now_seconds();
        multiply(A, B, C, n);
        double elapsed = now_seconds() - start;

        /* The checksum consumes the result, so the work cannot be skipped. */
        double checksum = 0.0;
        for (size_t x = 0; x < (size_t)n * n; x++) checksum += C[x];

        int is_warmup = r < warmups;
        printf("%s,%d,%.9f,%.17g\n", is_warmup ? "warmup" : "measured",
               is_warmup ? r + 1 : r - warmups + 1, elapsed, checksum);
        fflush(stdout);
    }

    if (argc >= 7) {
        FILE *out = fopen(argv[6], "wb");
        fwrite(C, sizeof(double), (size_t)n * n, out);
        fclose(out);
    }
    free(A);
    free(B);
    free(C);
    return 0;
}
