#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// Inicializa matrices aleatorias
void init_matrices(double *A, double *B, double *C_seq, double *C_par, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)(rand() % 100) / 10.0;
        B[i] = (double)(rand() % 100) / 10.0;
        C_seq[i] = 0.0;
        C_par[i] = 0.0;
    }
}

// 1. Baseline Secuencial (O(n^3))
void multiply_sequential(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// 2. Implementación Paralela con OpenMP
void multiply_parallel(double *A, double *B, double *C, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// 3. Validación de corrección de datos
int verify_results(double *C_seq, double *C_par, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(C_seq[i] - C_par[i]) > 1e-9) {
            return 0; // Falla la validación
        }
    }
    return 1; // Correcto
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <tamaño_n> <num_hilos>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C_seq = (double *)malloc(n * n * sizeof(double));
    double *C_par = (double *)malloc(n * n * sizeof(double));

    init_matrices(A, B, C_seq, C_par, n);

    // Medición Secuencial
    double start_seq = omp_get_wtime();
    multiply_sequential(A, B, C_seq, n);
    double end_seq = omp_get_wtime();
    double time_seq = end_seq - start_seq;

    // Medición Paralela
    double start_par = omp_get_wtime();
    multiply_parallel(A, B, C_par, n, num_threads);
    double end_par = omp_get_wtime();
    double time_par = end_par - start_par;

    // Validación
    int is_valid = verify_results(C_seq, C_par, n);

    // Salida formato CSV: N, Hilos, TiempoSec, TiempoPar, Valido
    printf("%d,%d,%f,%f,%d\n", n, num_threads, time_seq, time_par, is_valid);

    free(A); free(B); free(C_seq); free(C_par);
    return 0;
}