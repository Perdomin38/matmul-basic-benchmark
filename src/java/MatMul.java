/*
 * Basic dense matrix multiplication C = A x B (n x n, float64) in Java.
 *
 * Storage: double[n][n], i.e. an array of n row arrays. Each row is a
 * contiguous block of n doubles, but different rows are separate objects
 * on the JVM heap (not guaranteed to be adjacent in memory).
 * Algorithm: textbook triple loop in i-j-k order, O(n^3) multiply-adds.
 *
 * Usage: java MatMul <n> <A.bin> <B.bin> <warmups> <reps> [C_out.bin]
 *   Input files: n*n little-endian float64 values, row-major.
 *   Output (stdout): CSV lines "phase,run,time_s,checksum"
 *   If C_out.bin is given, the last result matrix is written there.
 *
 * Timed region: only the triple loop. Reading the inputs and allocating
 * the result happen before the timer starts.
 */
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

public class MatMul {

    static double[][] readMatrix(String path, int n) throws IOException {
        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(Paths.get(path)))
                .order(ByteOrder.LITTLE_ENDIAN);
        if (buf.capacity() != 8L * n * n) {
            throw new IOException("file " + path + " does not contain " + (n * n) + " values");
        }
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = buf.getDouble();
            }
        }
        return m;
    }

    static void writeMatrix(String path, double[][] m, int n) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * n * n).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                buf.putDouble(m[i][j]);
            }
        }
        Files.write(Paths.get(path), buf.array());
    }

    static void multiply(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 5) {
            System.err.println("usage: java MatMul n A.bin B.bin warmups reps [C_out.bin]");
            System.exit(1);
        }
        int n = Integer.parseInt(args[0]);
        int warmups = Integer.parseInt(args[3]);
        int reps = Integer.parseInt(args[4]);

        double[][] A = readMatrix(args[1], n);
        double[][] B = readMatrix(args[2], n);
        double[][] C = new double[n][n];

        System.out.println("phase,run,time_s,checksum");
        for (int r = 0; r < warmups + reps; r++) {
            long start = System.nanoTime();
            multiply(A, B, C, n);
            long elapsed = System.nanoTime() - start;

            // The checksum consumes the result, so the work cannot be skipped.
            double checksum = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    checksum += C[i][j];
                }
            }

            boolean isWarmup = r < warmups;
            System.out.printf(java.util.Locale.ROOT, "%s,%d,%.9f,%s%n",
                    isWarmup ? "warmup" : "measured",
                    isWarmup ? r + 1 : r - warmups + 1,
                    elapsed / 1e9, Double.toString(checksum));
            System.out.flush();
        }

        if (args.length >= 6) {
            writeMatrix(args[5], C, n);
        }
    }
}
