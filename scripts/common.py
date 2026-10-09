"""Shared configuration for every script. All fixed experimental settings live here."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
INPUT_DIR = ROOT / "data" / "inputs"
TEST_DIR = ROOT / "data" / "tests"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ---- Inputs -------------------------------------------------------------
SEED = 2026                     # numpy.random.default_rng seed
VALUE_RANGE = (-1.0, 1.0)       # uniform values in [-1, 1)
SIZES = [100, 200, 400, 800, 1600, 3200]   # declared size sequence (doubling)

# ---- Measurement protocol ------------------------------------------------
MEASURED_REPS = 5
WARMUPS = {"c": 1, "java": 2, "python": 1}
TIME_BUDGET_S = 600             # max wall time for one configuration (warm-ups + reps)
MEMORY_BUDGET_MB = 4096         # max peak memory for one configuration
MEMORY_SAMPLE_INTERVAL_S = 0.01
# The i7-13620H is a hybrid CPU: logical processors 0-11 belong to the 6 performance
# cores (P-cores) and 12-15 to the 4 efficiency cores (E-cores) (Windows
# GetSystemCpuSetInformation, EfficiencyClass). An exploratory run without pinning showed
# large run-to-run variation, so every benchmark process is restricted to the P-cores
# and given high priority. Set to None on another machine to disable pinning.
CPU_AFFINITY = list(range(12))
HIGH_PRIORITY = True

# ---- Toolchain (fixed settings) ------------------------------------------
CC = os.environ.get("CC", "gcc")
C_FLAGS = ["-O2", "-std=c11", "-static"]   # -static: no MinGW DLLs needed at run time
JAVAC = "javac"
JAVA = "java"
JVM_FLAGS = ["-Xmx3g"]   # heap cap below the memory budget
PYTHON = sys.executable

C_EXE = BUILD_DIR / ("matmul_c.exe" if os.name == "nt" else "matmul_c")
JAVA_CLASS_DIR = BUILD_DIR / "java"
PY_SCRIPT = ROOT / "src" / "python" / "matmul.py"

LANGUAGES = ["c", "java", "python"]


def command(lang, n, a_path, b_path, warmups, reps, out_path=None):
    """Return the command line that runs one implementation."""
    args = [str(n), str(a_path), str(b_path), str(warmups), str(reps)]
    if out_path is not None:
        args.append(str(out_path))
    if lang == "c":
        return [str(C_EXE)] + args
    if lang == "java":
        return [JAVA] + JVM_FLAGS + ["-cp", str(JAVA_CLASS_DIR), "MatMul"] + args
    if lang == "python":
        return [PYTHON, str(PY_SCRIPT)] + args
    raise ValueError(lang)


def input_paths(n):
    return INPUT_DIR / f"A_{n}.bin", INPUT_DIR / f"B_{n}.bin"
