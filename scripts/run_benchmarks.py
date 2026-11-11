import os
import subprocess
import numpy as np
import pandas as pd

# 1. Control estricto de multithreading implícito en librerías externas
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Configuración del experimento
SIZES = [250, 500, 750, 1000]
THREADS_LIST = [1, 2, 4, 6, 8, 12, 16]
REPETITIONS = 5

# Rutas según SO
EXECUTABLE = os.path.join("..", "src", "matrix_mult.exe" if os.name == "nt" else "matrix_mult")
OUTPUT_CSV = os.path.join("..", "results", "benchmark_results.csv")

def run_experiment():
    results = []
    
    print("Iniciando batería de pruebas...")
    print(f"Rutas de ejecutable: {EXECUTABLE}")
    
    for n in SIZES:
        for p in THREADS_LIST:
            seq_times = []
            par_times = []
            valid_flag = True

            for rep in range(REPETITIONS):
                try:
                    cmd = [EXECUTABLE, str(n), str(p)]
                    output = subprocess.check_output(cmd, text=True).strip()
                    parts = output.split(",")

                    t_seq = float(parts[2])
                    t_par = float(parts[3])
                    is_valid = int(parts[4]) == 1

                    seq_times.append(t_seq)
                    par_times.append(t_par)
                    if not is_valid:
                        valid_flag = False
                except Exception as e:
                    print(f"Error en N={n}, p={p}, rep={rep}: {e}")

            # Estadísticos de resumen (Mediana e IQR)
            seq_median = np.median(seq_times)
            seq_q75, seq_q25 = np.percentile(seq_times, [75 ,25])
            seq_iqr = seq_q75 - seq_q25

            par_median = np.median(par_times)
            par_q75, par_q25 = np.percentile(par_times, [75 ,25])
            par_iqr = par_q75 - par_q25

            # Métricas principales
            speedup = seq_median / par_median if par_median > 0 else 0
            efficiency = speedup / p

            results.append({
                "N": n,
                "Threads": p,
                "T_seq_median": seq_median,
                "T_seq_iqr": seq_iqr,
                "T_par_median": par_median,
                "T_par_iqr": par_iqr,
                "Speedup": speedup,
                "Efficiency": efficiency,
                "Valid": valid_flag
            })

            print(f"N={n:4d} | p={p:2d} | T_seq={seq_median:.4f}s | T_par={par_median:.4f}s | S(p)={speedup:.2f}x | E(p)={efficiency:.2f}")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResultados guardados exitosamente en: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_experiment()