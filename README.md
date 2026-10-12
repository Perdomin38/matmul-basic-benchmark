# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC, 2026-2027.
Author: Daniel Perdomo Medina.

**Research question:** how does the matrix size *n* affect the runtime and the
memory use of the same textbook O(n³) matrix multiplication implemented in
Python, Java and C, on one fixed machine?

The report source is `report/report.tex` (LaTeX, figures in `report/figures/`); it is compiled
to `report/Assignment01_PerdomoMedina_Report.pdf` with pdfLaTeX (e.g. on Overleaf).

## Repository layout

```
src/c/matmul.c          C implementation   (flat row-major double array)
src/java/MatMul.java    Java implementation (double[n][n], array of row arrays)
src/python/matmul.py    Python implementation (list of lists of float, no NumPy)
scripts/common.py       ALL fixed settings: sizes, seed, repetitions, warm-ups, budgets, flags
scripts/build.py        compiles C and Java, writes results/raw/environment.txt
scripts/generate_inputs.py  writes the shared input matrices data/inputs/A_n.bin, B_n.bin
scripts/verify.py       correctness tests (known answer, identity, zero, NumPy reference)
scripts/run_benchmarks.py   runs every (language, n) configuration, measures time and memory
scripts/analyze.py      builds tables (results/tables) and figures (results/figures)
results/raw/            raw measurements (CSV), correctness results, environment, log
results/tables/         summary tables (CSV and Markdown) produced by analyze.py
results/figures/        figures produced by analyze.py
results/exploratory_unpinned/  first run without CPU pinning (kept, not used in the analysis)
report/                 LaTeX source of the report and the figures it uses
```

All three programs share the same command-line interface:

```
<program> <n> <A.bin> <B.bin> <warmups> <reps> [C_out.bin]
```

Input files hold n·n little-endian float64 values in row-major order. Each program
prints one CSV line per run: `phase,run,time_s,checksum`.

Raw data columns (`results/raw/`):
- `benchmark_raw.csv`: `language, n, phase (warmup/measured), run, time_s, checksum, checksum_ok`
- `runs_status.csv`: `language, n, warmups, reps, status, wall_time_s, peak_memory_mb`

Every configuration uses float64 values and runs single-threaded (one thread per process,
no numerical library inside the kernel), so these two labels are not repeated in each row.

## Requirements (versions used)

| Tool | Version |
|---|---|
| OS / CPU | Windows 11 Home (build 26200), Intel Core i7-13620H, 16 GB RAM |
| C compiler | GCC 16.2.0 (MinGW-w64 UCRT, WinLibs), flags `-O2 -std=c11 -static` |
| Java | Oracle JDK 24.0.1 (HotSpot), run with `-Xmx3g` |
| Python | CPython 3.11.9 |
| Helper packages | see `requirements.txt` (NumPy, pandas, matplotlib, psutil) |

`gcc`, `javac`/`java` and `python` must be on the `PATH`.
On Windows, GCC can be installed with `winget install BrechtSanders.WinLibs.POSIX.UCRT`.

## How to reproduce

Run from the `scripts/` directory:

```bash
pip install -r ../requirements.txt
python build.py            # compile C and Java, record environment
python generate_inputs.py  # create data/inputs (seed 2026, values in [-1, 1))
python verify.py           # correctness tests; must print ALL TESTS PASSED
python run_benchmarks.py   # full benchmark (about 30-45 min on the test machine)
python analyze.py          # tables and figures from the raw CSV files
```

The input matrices (`data/inputs/*.bin`, about 210 MB) and the compiled programs (`build/`)
are not stored in the repository; `generate_inputs.py` recreates exactly the same inputs
because the seed is fixed, and `build.py` recompiles the programs.

`run_benchmarks.py` accepts `--langs` and `--sizes` for a quick partial run. Without
`--output-dir` it overwrites `results/raw/benchmark_raw.csv` and `runs_status.csv`, so a
quick test should write elsewhere:
`python run_benchmarks.py --sizes 100 200 --output-dir ../results/test_run`.


## Experimental protocol (summary)

- Sizes n = 100, 200, 400, 800, 1600, 3200 (doubling, so cubic growth means ×8 per step).
- Same input files for all languages; float64 everywhere; same i-j-k loop order.
- Timed region: the triple loop only (input reading and result allocation are excluded).
- Warm-up runs: C 1, Java 2 (JIT compilation), Python 1; then 5 measured runs.
- Reported: median and IQR of the 5 measured runs; every raw run is kept.
- Each benchmark process is pinned to the 6 performance cores (logical CPUs 0-11) and runs
  at high priority. The first, unpinned run is kept in `results/exploratory_unpinned/`.
- Memory: peak working set (peak resident memory) of the whole process, sampled with psutil.
- Budgets: 600 s wall time and 4096 MB per configuration; larger sizes are skipped after a failure.
- Correctness: every measured run's checksum is compared with NumPy's result.

## AI usage

An AI assistant (Claude, Anthropic) was used to help plan the experiment, write and
debug the scripts and revise the English of the report. All measurements were produced
by running the code in this repository on the machine described above.
