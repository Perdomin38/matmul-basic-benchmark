"""Build the summary tables and figures of the report from the raw CSV files.

Inputs : results/raw/benchmark_raw.csv, results/raw/runs_status.csv
Outputs: results/tables/*.csv and *.md, results/figures/*.png

Only measured runs (not warm-ups) of configurations with status OK are summarised.
Statistics per (language, n): number of repetitions, median, first and third
quartiles, interquartile range (IQR = Q3 - Q1), minimum and maximum."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES_DIR, LANGUAGES, RAW_DIR, TABLES_DIR

LABELS = {"c": "C (gcc -O2)", "java": "Java (HotSpot JIT)", "python": "Python (CPython)"}
COLORS = {"c": "#1f77b4", "java": "#d62728", "python": "#2ca02c"}
MARKERS = {"c": "o", "java": "s", "python": "^"}


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_table(df, name):
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    (TABLES_DIR / f"{name}.md").write_text(to_markdown(df), encoding="utf-8")


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_DIR / "benchmark_raw.csv")
    status = pd.read_csv(RAW_DIR / "runs_status.csv")

    ok = status[status.status == "OK"][["language", "n", "peak_memory_mb"]]
    measured = raw[raw.phase == "measured"].merge(ok, on=["language", "n"])
    assert measured.checksum_ok.all(), "a checksum differs from the NumPy reference"

    # ---- Table 1: runtime statistics -------------------------------------
    g = measured.groupby(["language", "n"])["time_s"]
    summary = pd.DataFrame({
        "reps": g.count(),
        "median_s": g.median(),
        "q1_s": g.quantile(0.25),
        "q3_s": g.quantile(0.75),
        "min_s": g.min(),
        "max_s": g.max(),
    }).reset_index()
    summary["iqr_s"] = summary.q3_s - summary.q1_s
    summary["iqr_pct_of_median"] = 100 * summary.iqr_s / summary.median_s
    summary["ns_per_multiply_add"] = 1e9 * summary.median_s / summary.n.astype(float) ** 3
    summary["growth_vs_previous_n"] = summary.groupby("language").median_s.transform(
        lambda s: s / s.shift(1))
    summary = summary.merge(ok, on=["language", "n"])
    summary["array_storage_mb"] = 3 * 8 * summary.n.astype(float) ** 2 / 2**20
    summary["language"] = pd.Categorical(summary.language, LANGUAGES, ordered=True)
    summary = summary.sort_values(["language", "n"])
    summary.to_csv(TABLES_DIR / "summary_full.csv", index=False)

    runtime = pd.DataFrame({
        "Language": summary.language.astype(str),
        "n": summary.n,
        "Reps": summary.reps,
        "Median time (s)": summary.median_s.map(lambda v: f"{v:.4g}"),
        "IQR (s)": summary.iqr_s.map(lambda v: f"{v:.2g}"),
        "IQR / median (%)": summary.iqr_pct_of_median.map(lambda v: f"{v:.1f}"),
        "Min-max (s)": [f"{a:.4g}-{b:.4g}" for a, b in zip(summary.min_s, summary.max_s)],
        "ns per multiply-add": summary.ns_per_multiply_add.map(lambda v: f"{v:.2f}"),
        "Growth vs n/2": summary.growth_vs_previous_n.map(
            lambda v: "-" if pd.isna(v) else f"{v:.1f}x"),
    })
    save_table(runtime, "runtime_summary")

    # ---- Table 2: memory --------------------------------------------------
    memory = pd.DataFrame({
        "Language": summary.language.astype(str),
        "n": summary.n,
        "Peak process memory (MB)": summary.peak_memory_mb.map(lambda v: f"{v:.1f}"),
        "3 x 8n^2 array estimate (MB)": summary.array_storage_mb.map(lambda v: f"{v:.2f}"),
    })
    save_table(memory, "memory_summary")

    # ---- Table 3: ratios relative to C ------------------------------------
    pivot = summary.pivot(index="n", columns="language", values="median_s")
    ratios = pd.DataFrame({"n": pivot.index})
    for lang in ["java", "python"]:
        ratios[f"{lang} / C"] = [("-" if pd.isna(v) else f"{v:.1f}")
                                 for v in (pivot[lang] / pivot["c"]).values]
    save_table(ratios, "ratio_to_c")

    # ---- Table 4: log-log slope (empirical growth exponent) ---------------
    slopes = []
    for lang in LANGUAGES:
        s = summary[summary.language == lang]
        if len(s) >= 2:
            slope = np.polyfit(np.log(s.n.astype(float)), np.log(s.median_s), 1)[0]
            slopes.append({"Language": lang, "Sizes used": ", ".join(map(str, s.n)),
                           "Fitted exponent b in t = a n^b": f"{slope:.2f}"})
    save_table(pd.DataFrame(slopes), "growth_exponent")

    # ---- Table 5: limits (status of every configuration) ------------------
    limits = status[["language", "n", "status", "wall_time_s", "peak_memory_mb"]].fillna("-")
    limits.columns = ["Language", "n", "Status", "Wall time incl. warm-up (s)",
                      "Peak memory (MB)"]
    save_table(limits, "limits")

    # ---- Table 6: warm-up effect -------------------------------------------
    warm = raw[raw.phase == "warmup"].merge(ok[["language", "n"]], on=["language", "n"])
    warm = warm.groupby(["language", "n"]).time_s.first().rename("first_warmup_s")
    warm = warm.reset_index().merge(summary[["language", "n", "median_s"]].astype(
        {"language": str}), on=["language", "n"])
    warm["first warm-up / measured median"] = (warm.first_warmup_s / warm.median_s).map(
        lambda v: f"{v:.2f}")
    warm["first_warmup_s"] = warm.first_warmup_s.map(lambda v: f"{v:.4g}")
    warm["median_s"] = warm.median_s.map(lambda v: f"{v:.4g}")
    warm.columns = ["Language", "n", "First warm-up (s)", "Measured median (s)",
                    "First warm-up / measured median"]
    save_table(warm, "warmup_effect")

    # ---- Figure 1: runtime vs n (log-log) with cubic reference -------------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for lang in LANGUAGES:
        s = summary[summary.language == lang]
        yerr = [s.median_s - s.q1_s, s.q3_s - s.median_s]
        ax.errorbar(s.n, s.median_s, yerr=yerr, marker=MARKERS[lang], color=COLORS[lang],
                    capsize=3, label=LABELS[lang])
        n0, t0 = s.n.iloc[0], s.median_s.iloc[0]
        ref_n = np.array([n0, s.n.iloc[-1]], dtype=float)
        ax.plot(ref_n, t0 * (ref_n / n0) ** 3, ":", color=COLORS[lang], alpha=0.6)
    ax.plot([], [], ":", color="grey", label="cubic model t ~ n^3 (from smallest n)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(summary.n.unique()))
    ax.set_xticklabels(sorted(summary.n.unique()))
    ax.set_xlabel("Matrix size n (n x n matrices)")
    ax.set_ylabel("Median multiplication time (s), log scale")
    ax.set_title("Runtime vs matrix size (median of 5 runs, bars = IQR)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_runtime_vs_n.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: time per multiply-add ----------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for lang in LANGUAGES:
        s = summary[summary.language == lang]
        ax.plot(s.n, s.ns_per_multiply_add, marker=MARKERS[lang], color=COLORS[lang],
                label=LABELS[lang])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(summary.n.unique()))
    ax.set_xticklabels(sorted(summary.n.unique()))
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Median time / n^3 (ns per multiply-add), log scale")
    ax.set_title("Cost per inner-loop operation (flat line = perfect cubic scaling)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_time_per_operation.png", dpi=150)
    plt.close(fig)

    # ---- Figure 3: peak memory vs n ----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for lang in LANGUAGES:
        s = summary[summary.language == lang]
        ax.plot(s.n, s.peak_memory_mb, marker=MARKERS[lang], color=COLORS[lang],
                label=LABELS[lang])
    ns = np.array(sorted(summary.n.unique()), dtype=float)
    ax.plot(ns, 3 * 8 * ns**2 / 2**20, "k--", label="3 x 8n^2 bytes (raw float64 arrays)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([int(v) for v in ns])
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Peak process memory (MB = 2^20 bytes), log scale")
    ax.set_title("Peak resident memory of the whole process (psutil, peak working set)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_memory_vs_n.png", dpi=150)
    plt.close(fig)

    print(runtime.to_string(index=False))
    print()
    print(memory.to_string(index=False))
    print()
    print(ratios.to_string(index=False))
    print()
    print(pd.DataFrame(slopes).to_string(index=False))
    print()
    print(limits.to_string(index=False))
    print()
    print(warm.to_string(index=False))


if __name__ == "__main__":
    main()
