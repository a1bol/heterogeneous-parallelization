#include <iostream>
#include <vector>
#include <mpi.h>
#include <iomanip>
#include <cmath>
#include <cstdlib>

using namespace std;

/**
 * ЗАДАНИЕ 2: Распределённое решение системы линейных уравнений методом Гаусса.
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 4;
    if (argc > 1) N = atoi(argv[1]);

    if (N % size != 0) {
        if (rank == 0) cerr << "Error: N must be divisible by number of processes." << endl;
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / size;
    vector<double> matrix;
    vector<double> local_matrix(rows_per_proc * (N + 1));

    if (rank == 0) {
        cout << "=========================================================" << endl;
        cout << "Task 2: Distributed Gauss Method" << endl;
        cout << "Matrix size: " << N << "x" << N << " | Processes: " << size << endl;
        cout << "=========================================================" << endl;

        matrix.resize(N * (N + 1));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N + 1; j++) {
                matrix[i * (N + 1) + j] = (rand() % 10) + 1.0;
            }
        }
    }

    double start_time = MPI_Wtime();

    MPI_Scatter(matrix.data(), rows_per_proc * (N + 1), MPI_DOUBLE,
                local_matrix.data(), rows_per_proc * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> pivot_row(N + 1);
    for (int k = 0; k < N; k++) {
        int root = k / rows_per_proc;

        if (rank == root) {
            int local_k = k % rows_per_proc;
            double pivot = local_matrix[local_k * (N + 1) + k];
            for (int j = k; j < N + 1; j++) {
                local_matrix[local_k * (N + 1) + j] /= pivot;
                pivot_row[j] = local_matrix[local_k * (N + 1) + j];
            }
        }

        MPI_Bcast(pivot_row.data() + k, (N + 1) - k, MPI_DOUBLE, root, MPI_COMM_WORLD);

        for (int i = 0; i < rows_per_proc; i++) {
            int global_i = rank * rows_per_proc + i;
            if (global_i > k) {
                double factor = local_matrix[i * (N + 1) + k];
                for (int j = k; j < N + 1; j++) {
                    local_matrix[i * (N + 1) + j] -= factor * pivot_row[j];
                }
            }
        }
    }

    MPI_Gather(local_matrix.data(), rows_per_proc * (N + 1), MPI_DOUBLE,
               matrix.data(), rows_per_proc * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        vector<double> x(N);
        for (int i = N - 1; i >= 0; i--) {
            x[i] = matrix[i * (N + 1) + N];
            for (int j = i + 1; j < N; j++) {
                x[i] -= matrix[i * (N + 1) + j] * x[j];
            }
        }

        double end_time = MPI_Wtime();
        cout << "Solution found. Top 5 elements:" << endl;
        for (int i = 0; i < min(N, 5); i++) {
            cout << "x[" << i << "] = " << x[i] << endl;
        }
        cout << "Execution time: " << end_time - start_time << " seconds" << endl;
        cout << "=========================================================" << endl;
    }

    MPI_Finalize();
    return 0;
}
