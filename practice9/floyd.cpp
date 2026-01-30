#include <iostream>
#include <vector>
#include <mpi.h>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

using namespace std;

#define INF 10000

/**
 * ЗАДАНИЕ 3: Параллельный анализ графов (алгоритм Флойда-Уоршелла).
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 300;
    if (argc > 1) N = atoi(argv[1]);

    int rows_per_proc = N / size;
    if (N % size != 0) {
         if (rank == 0) cerr << "Error: N must be divisible by size." << endl;
         MPI_Finalize();
         return 1;
    }

    vector<int> matrix;
    vector<int> local_matrix(rows_per_proc * N);

    if (rank == 0) {
        cout << "=========================================================" << endl;
        cout << "Task 3: Parallel Floyd-Warshall Algorithm" << endl;
        cout << "Graph size: " << N << "x" << N << " | Processes: " << size << endl;
        cout << "=========================================================" << endl;

        matrix.resize(N * N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) matrix[i * N + j] = 0;
                else matrix[i * N + j] = (rand() % 50 == 0) ? INF : (rand() % 20 + 1);
            }
        }
    }

    double start_time = MPI_Wtime();

    MPI_Scatter(matrix.data(), rows_per_proc * N, MPI_INT,
                local_matrix.data(), rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> row_k(N);

    for (int k = 0; k < N; k++) {
        int root = k / rows_per_proc;

        if (rank == root) {
            int local_k = k % rows_per_proc;
            for (int j = 0; j < N; j++) {
                row_k[j] = local_matrix[local_k * N + j];
            }
        }

        MPI_Bcast(row_k.data(), N, MPI_INT, root, MPI_COMM_WORLD);

        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < N; j++) {
                if (local_matrix[i * N + k] + row_k[j] < local_matrix[i * N + j]) {
                    local_matrix[i * N + j] = local_matrix[i * N + k] + row_k[j];
                }
            }
        }
    }

    MPI_Gather(local_matrix.data(), rows_per_proc * N, MPI_INT,
               matrix.data(), rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        cout << "Computation finished." << endl;
        cout << "Execution time: " << end_time - start_time << " seconds" << endl;
        cout << "=========================================================" << endl;
        
        cout << "Sample of result (Top 5x5 matrix):" << endl;
        for (int i = 0; i < min(N, 5); i++) {
            for (int j = 0; j < min(N, 5); j++) {
                if (matrix[i * N + j] >= INF) cout << "INF ";
                else cout << setw(3) << matrix[i * N + j] << " ";
            }
            cout << endl;
        }
        cout << "=========================================================" << endl;
    }

    MPI_Finalize();
    return 0;
}
