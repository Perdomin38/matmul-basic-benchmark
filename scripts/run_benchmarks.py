"""Run the benchmark for every (language, n) configuration.

One configuration = one process that loads A_n and B_n, runs the warm-up
multiplications and then MEASURED_REPS measured multiplications. Each program
times only its own kernel (monotonic timer) and prints one CSV line per run.
Each process is pinned to the P-cores and runs at high priority (see common.py).

Memory: while the process runs, this script samples it with psutil every
10 ms. The metric is the PEAK RESIDENT MEMORY of the whole process
(Windows: peak working set "peak_wset"; elsewhere: maximum sampled RSS).
It therefore includes the language runtime (interpreter / JVM), not only the
matrices.

Limits: a configuration is killed and recorded as TIMEOUT if it exceeds
TIME_BUDGET_S of wall time, or MEMORY_LIMIT if its memory exceeds
MEMORY_BUDGET_MB. Larger sizes of that language are then recorded as SKIPPED
(they would cost at least 8x more). No execution time is invented for them.

Outputs (results/raw/):
  benchmark_raw.csv  every warm-up and measured run
  runs_status.csv    one row per configuration: status, wall time, peak memory

Usage: python run_benchmarks.py [--langs c java python] [--sizes 100 200 ...] [--output-dir DIR]
  --output-dir writes the CSV files to another folder (default results/raw), e.g. for a
  quick test that must not overwrite the official results."""
import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil

from common import (CPU_AFFINITY, HIGH_PRIORITY, LANGUAGES, MEASURED_REPS, MEMORY_BUDGET_MB, MEMORY_SAMPLE_INTERVAL_S,
                    RAW_DIR, SIZES, TIME_BUDGET_S, WARMUPS, command, input_paths)


def memory_mb(proc):
    info = proc.memory_info()
    return getattr(info, "peak_wset", info.rss) / 2**20


def run_configuration(cmd):
    """Run one process, return (status, stdout, wall_time_s, peak_memory_mb)."""
    start = time.perf_counter()
    proc = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Pinning happens while the program is still loading its inputs (before any timing).
    if CPU_AFFINITY is not None:
        proc.cpu_affinity(CPU_AFFINITY)
    if HIGH_PRIORITY:
        proc.nice(psutil.HIGH_PRIORITY_CLASS if psutil.WINDOWS else -10)
    peak = 0.0
    status = "OK"
    while proc.poll() is None:
        try:
            peak = max(peak, memory_mb(proc))
        except psutil.Error:
            pass  # process ended between poll() and the memory read
        if time.perf_counter() - start > TIME_BUDGET_S:
            status = "TIMEOUT"
        elif peak > MEMORY_BUDGET_MB:
            status = "MEMORY_LIMIT"
        if status != "OK":
            proc.kill()
            break
        time.sleep(MEMORY_SAMPLE_INTERVAL_S)
    stdout, stderr = proc.communicate()
    wall = time.perf_counter() - start
    if status == "OK" and proc.returncode != 0:
        status = f"ERROR({proc.returncode}): {stderr.strip()[:200]}"
    return status, stdout, wall, peak


def reference_checksum(n):
    a_path, b_path = input_paths(n)
    A = np.fromfile(a_path, dtype="<f8").reshape(n, n)
    B = np.fromfile(b_path, dtype="<f8").reshape(n, n)
    return float(np.sum(A @ B))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=LANGUAGES)
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, status_rows = [], []
    for lang in args.langs:
        stopped = False
        for n in args.sizes:
            row = {"language": lang, "n": n, "warmups": WARMUPS[lang], "reps": MEASURED_REPS}
            if stopped:
                status_rows.append({**row, "status": "SKIPPED", "wall_time_s": "",
                                    "peak_memory_mb": ""})
                print(f"{lang:6s} n={n:5d} SKIPPED (a smaller size exceeded a budget)")
                continue
            a_path, b_path = input_paths(n)
            cmd = command(lang, n, a_path, b_path, WARMUPS[lang], MEASURED_REPS)
            print(f"{lang:6s} n={n:5d} running ...", flush=True)
            status, stdout, wall, peak = run_configuration(cmd)

            ref = reference_checksum(n)
            for line in stdout.strip().splitlines()[1:]:
                phase, run, t, checksum = line.split(",")
                ok = abs(float(checksum) - ref) <= 1e-9 * max(1.0, abs(ref))
                raw_rows.append({"language": lang, "n": n, "phase": phase, "run": int(run),
                                 "time_s": float(t), "checksum": checksum,
                                 "checksum_ok": ok})
            status_rows.append({**row, "status": status, "wall_time_s": round(wall, 3),
                                "peak_memory_mb": round(peak, 2)})
            print(f"{lang:6s} n={n:5d} {status} wall={wall:.1f}s peak_mem={peak:.1f}MB",
                  flush=True)
            if status != "OK":
                stopped = True

    with open(out_dir / "benchmark_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "n", "phase", "run", "time_s",
                                               "checksum", "checksum_ok"])
        writer.writeheader()
        writer.writerows(raw_rows)
    with open(out_dir / "runs_status.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "n", "warmups", "reps", "status",
                                               "wall_time_s", "peak_memory_mb"])
        writer.writeheader()
        writer.writerows(status_rows)


if __name__ == "__main__":
    main()
