# Assignment 3: Shared-Memory Parallel Matrix Multiplication and SIMD Analysis

**Author:** Alejandro Morales Monroy
**Degree:** Data Science and Engineering (ULPGC)  
**Course:** Big Data  

## Overview
This repository contains the benchmark infrastructure and technical report for investigating thread scalability, framework overhead, hardware bounds, and compiler SIMD vectorization in a shared-memory matrix multiplication algorithm ($O(n^3)$).

## Repository Structure
```text
assignment3_matrix_parallel/
├── README.md                          # Project documentation
├── .gitignore                         # Build and environment exclusions
├── data/                              # Data output directory
├── results/                           # Benchmark CSVs and generated plots
│   ├── benchmark_results.csv          # Parallel scalability raw data
│   ├── simd_comparison_results.csv    # SIMD vs. Non-SIMD comparison data
│   ├── speedup_plot.png               # Speedup S(p) vs. Thread count plot
│   ├── efficiency_plot.png            # Efficiency E(p) vs. Thread count plot
│   └── time_variability_plot.png      # Runtime & IQR variability plot (N=1000)
├── scripts/                           # Python automation scripts
│   ├── run_benchmarks.py              # Parallel execution pipeline
│   ├── run_simd_benchmark.py          # SIMD evaluation pipeline
│   └── plot_results.py                # Visualization generator
└── src/                               # C source code & executables
    ├── matrix_mult.c                  # Core sequential, OpenMP & validation source
    ├── matrix_mult.exe                # Non-vectorized parallel executable
    └── matrix_mult_simd.exe           # AVX2 vectorized parallel executable
```

## Hardware & Enviroment Specification
* **CPU**: 6 Physical Cores / 12 Logical Processors (Intel Core i7-10750H)
* **Compiler**: GCC 16.2.0 (WinLibs MinGW-w64)
* **Operating System**: Windows 11
* **Threading Library**: OpenMP (`-fopenmp`)

## Building the binaries
From the `src\` directory, compile both non-vectorized and vectorized binaries:
```
# Non-vectorized parallel binary (Baseline & Thread Scalability)
gcc -O2 -fno-tree-vectorize -fopenmp matrix_mult.c -o matrix_mult.exe

# SIMD-vectorized parallel binary (AVX2 / Native ISA)
gcc -O3 -march=native -ftree-vectorize -fopenmp -fopt-info-vec-optimized matrix_mult.c -o matrix_mult_simd.exe
```

## Running the Benchmarks
**1.** Install Python dependencies:
```
pip install numpy pandas matplotlib
```

**2.** Execute thread scalability benchmarks (disables implicit multithreading in external libraries automatically):
```
cd scripts
python run_benchmarks.py
```

**3.** Execute SIMD comparative benchmarks:
```
python run_simd_benchmark.py
```

**4.** Generate visualization plots:
```
python plot_results.py
```

## Key findings

- Framework Overhead ($p=1$): $<0.5\%$ overhead compared to pure sequential execution.
- Physical Core Limit ($p \le 6$): Near-linear scaling bound by memory bandwidth due to non-contiguous matrix accesses.
- Hyper-Threading ($6 < p \le 12$): Negligible runtime gains; efficiency drops significantly ($E(12) \approx 0.20$).
- Oversubscription ($p = 16$): Increased context switching and barrier contention degrade overall throughput.
- SIMD Impact: Speedup remains modest ($\approx 1.05\times$) because inner-loop matrix multiplication is memory-bound rather than compute-bound.