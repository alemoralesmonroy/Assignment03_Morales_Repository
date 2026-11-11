import os
import subprocess
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

SIZES = [250, 500, 750, 1000]
THREADS_TO_TEST = [1, 6]
REPETITIONS = 5

EXE_NO_SIMD = os.path.join("..", "src", "matrix_mult.exe")
EXE_SIMD = os.path.join("..", "src", "matrix_mult_simd.exe")

results = []

print("Running SIMD vs Non-SIMD Comparative Benchmark...")

for n in SIZES:
    for p in THREADS_TO_TEST:
        t_no_simd_list = []
        for _ in range(REPETITIONS):
            out = subprocess.check_output([EXE_NO_SIMD, str(n), str(p)], text=True).strip()
            t_no_simd_list.append(float(out.split(",")[3]))
        med_no_simd = np.median(t_no_simd_list)

        t_simd_list = []
        for _ in range(REPETITIONS):
            out = subprocess.check_output([EXE_SIMD, str(n), str(p)], text=True).strip()
            t_simd_list.append(float(out.split(",")[3]))
        med_simd = np.median(t_simd_list)

        simd_speedup = med_no_simd / med_simd if med_simd > 0 else 0

        results.append({
            "N": n,
            "Threads": p,
            "T_No_SIMD": med_no_simd,
            "T_SIMD": med_simd,
            "SIMD_Speedup": simd_speedup
        })

        print(f"N={n:4d} | p={p:2d} | No-SIMD: {med_no_simd:.4f}s | SIMD: {med_simd:.4f}s | Speedup SIMD: {simd_speedup:.2f}x")

df = pd.DataFrame(results)
output_path = os.path.join("..", "results", "simd_comparison_results.csv")
df.to_csv(output_path, index=False)
print(f"\nSIMD comparison saved to {output_path}")