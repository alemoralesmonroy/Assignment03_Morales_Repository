import os
import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark results
CSV_PATH = os.path.join("..", "results", "benchmark_results.csv")
RESULTS_DIR = os.path.join("..", "results")

df = pd.read_csv(CSV_PATH)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sizes = df["N"].unique()

# 1. Speedup S(p) Plot
plt.figure(figsize=(10, 6))
for n in sizes:
    sub_df = df[df["N"] == n]
    plt.plot(sub_df["Threads"], sub_df["Speedup"], marker='o', linewidth=2, label=f"N = {n}")

max_p = df["Threads"].max()
plt.plot([1, max_p], [1, max_p], 'k--', alpha=0.6, label="Ideal Speedup (Linear)")
plt.axvline(x=6, color='red', linestyle=':', alpha=0.7, label="Physical Cores (6)")
plt.axvline(x=12, color='orange', linestyle=':', alpha=0.7, label="Logical Processors (12)")

plt.title("Speedup S(p) vs. Thread Count (p)", fontsize=14, fontweight='bold')
plt.xlabel("Number of Threads (p)", fontsize=12)
plt.ylabel("Speedup S(p)", fontsize=12)
plt.xticks(df["Threads"].unique())
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "speedup_plot.png"), dpi=300)
plt.close()

# 2. Efficiency E(p) Plot
plt.figure(figsize=(10, 6))
for n in sizes:
    sub_df = df[df["N"] == n]
    plt.plot(sub_df["Threads"], sub_df["Efficiency"], marker='s', linewidth=2, label=f"N = {n}")

plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.6, label="Ideal Efficiency (1.0)")
plt.axvline(x=6, color='red', linestyle=':', alpha=0.7, label="Physical Cores (6)")
plt.axvline(x=12, color='orange', linestyle=':', alpha=0.7, label="Logical Processors (12)")

plt.title("Efficiency E(p) vs. Thread Count (p)", fontsize=14, fontweight='bold')
plt.xlabel("Number of Threads (p)", fontsize=12)
plt.ylabel("Efficiency E(p)", fontsize=12)
plt.xticks(df["Threads"].unique())
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "efficiency_plot.png"), dpi=300)
plt.close()

# 3. Execution Time & Variability (IQR) for N = 1000
plt.figure(figsize=(10, 6))
sub_1000 = df[df["N"] == 1000]

plt.errorbar(
    sub_1000["Threads"], 
    sub_1000["T_par_median"], 
    yerr=sub_1000["T_par_iqr"], 
    fmt='-o', 
    capsize=5, 
    capthick=2, 
    color='b', 
    ecolor='gray', 
    label="Parallel Time (Median ± IQR)"
)

t_seq_1000 = sub_1000["T_seq_median"].iloc[0]
plt.axhline(y=t_seq_1000, color='r', linestyle='--', label=f"Sequential Baseline ({t_seq_1000:.3f}s)")

plt.title("Execution Time and Variability (IQR) for N = 1000", fontsize=14, fontweight='bold')
plt.xlabel("Number of Threads (p)", fontsize=12)
plt.ylabel("Execution Time (s)", fontsize=12)
plt.xticks(df["Threads"].unique())
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "time_variability_plot.png"), dpi=300)
plt.close()

print("Plots successfully saved in English in the results/ directory.")