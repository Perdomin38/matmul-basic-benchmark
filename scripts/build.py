"""Compile the C and Java implementations with the fixed flags from common.py
and record the environment (hardware, OS, tool versions) in results/raw/environment.txt."""
import platform
import subprocess

import psutil

from common import (BUILD_DIR, CC, C_EXE, C_FLAGS, JAVA, JAVA_CLASS_DIR, JAVAC,
                    JVM_FLAGS, PYTHON, RAW_DIR, ROOT)


def run(cmd):
    # Run from the project root with relative paths: GCC's linker cannot open
    # absolute paths with non-ASCII characters (e.g. "Teoría") on Windows.
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def version(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def cpu_name():
    if platform.system() == "Windows":
        return version(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor).Name"])
    return platform.processor()


def main():
    BUILD_DIR.mkdir(exist_ok=True)
    JAVA_CLASS_DIR.mkdir(parents=True, exist_ok=True)
    run([CC] + C_FLAGS + ["-o", rel(C_EXE), "src/c/matmul.c"])
    run([JAVAC, "-d", rel(JAVA_CLASS_DIR), "src/java/MatMul.java"])

    import numpy
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"CPU: {cpu_name()}",
        f"Physical cores: {psutil.cpu_count(logical=False)}",
        f"Logical processors: {psutil.cpu_count(logical=True)}",
        f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB",
        f"OS: {platform.platform()}",
        "",
        f"C compiler: {version([CC, '--version']).splitlines()[0]}",
        f"C flags: {' '.join(C_FLAGS)}",
        f"Java: {version([JAVA, '-version']).splitlines()[0]} / "
        f"{version([JAVA, '-version']).splitlines()[-1]}",
        f"JVM flags: {' '.join(JVM_FLAGS)}",
        f"Python: {platform.python_implementation()} {platform.python_version()} ({PYTHON})",
        f"NumPy (inputs and reference only): {numpy.__version__}",
        f"psutil (memory measurement): {psutil.__version__}",
        "",
        "Threads: all three implementations are single-threaded; no numerical library",
        "is used inside the measured kernels.",
    ]
    (RAW_DIR / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
