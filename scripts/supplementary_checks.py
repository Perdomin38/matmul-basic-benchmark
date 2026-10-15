"""Supplementary checks discussed in the report (Section 5). They are not part of the
main benchmark and their results are not used in the summary tables.

  sustained  Python at n = 800, four processes back to back with no pause, alternating
             without / with CPU pinning and high priority; 0 warm-ups, 1 measured run each.
             Question: does the time grow under sustained load, independently of pinning?
  cooldown   C and Java at n = 1600, four processes in the order Java, C, C, Java, each
             started after a 60 s idle pause; same warm-ups as the main benchmark,
             C 3 and Java 2 measured runs. Question: is Java's jump at n = 1600 caused by
             the fixed run order of the main benchmark (Java ran right after 600 s of C)?

Each process is run with run_configuration() from run_benchmarks.py; every warm-up and
measured run is written to <output-dir>/supplementary_<check>.csv.

Usage: python supplementary_checks.py sustained|cooldown [--n N] [--pause S] [--output-dir DIR]
"""
import argparse
import csv
import time
from pathlib import Path

from common import RAW_DIR, WARMUPS, command, input_paths
from run_benchmarks import run_configuration

CHECKS = {
    # check: (default n, default pause in s, [(language, pinned, warm-ups, measured runs)])
    "sustained": (800, 0, [("python", False, 0, 1), ("python", True, 0, 1),
                           ("python", False, 0, 1), ("python", True, 0, 1)]),
    "cooldown": (1600, 60, [("java", True, WARMUPS["java"], 2), ("c", True, WARMUPS["c"], 3),
                            ("c", True, WARMUPS["c"], 3), ("java", True, WARMUPS["java"], 2)]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=CHECKS)
    parser.add_argument("--n", type=int)
    parser.add_argument("--pause", type=float, help="idle seconds before each process")
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    default_n, default_pause, processes = CHECKS[args.check]
    n = args.n or default_n
    pause = default_pause if args.pause is None else args.pause
    a_path, b_path = input_paths(n)
    rows = []
    for order, (lang, pinned, warmups, reps) in enumerate(processes, start=1):
        time.sleep(pause)
        status, stdout, wall, _ = run_configuration(
            command(lang, n, a_path, b_path, warmups, reps), pin=pinned)
        setting = "pinned" if pinned else "unpinned"
        print(f"{order} {lang:6s} n={n} {setting} {status} wall={wall:.1f}s", flush=True)
        for line in stdout.strip().splitlines()[1:]:
            phase, run, t, checksum = line.split(",")
            rows.append({"order": order, "language": lang, "n": n, "setting": setting,
                         "phase": phase, "run": int(run), "time_s": float(t),
                         "checksum": checksum})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"supplementary_{args.check}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written {out}")


if __name__ == "__main__":
    main()
