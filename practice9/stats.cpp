#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <mpi.h>
#include <ctime>

using namespace std;

/**
 * ЗАДАНИЕ 1: Распределённое вычисление среднего значения и стандартного отклонения.
 * Используется MPI_Scatterv для разделения массива с учётом остатка.
 */

int main(int argc, char** argv) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Получаем номер текущего процесса и общее их количество
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000; // Размер массива
    vector<int> data;
    
    // Подготовка параметров для MPI_Scatterv
    vector<int> sendcounts(size); // Количество элементов для каждого процесса
    vector<int> displs(size);    // Смещения в исходном массиве

    if (rank == 0) {
        // Процесс 0 создает массив случайных чисел
        data.resize(N);
        srand(static_cast<unsigned int>(time(0)));
        for (int i = 0; i < N; i++) {
            data[i] = rand() % 100; // Случайные числа от 0 до 99
        }

        // Вычисляем, сколько элементов отправить каждому процессу
        int rem = N % size; // Остаток, который нужно распределить
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = N / size + (i < rem ? 1 : 0);
            displs[i] = sum;
            sum += sendcounts[i];
        }
        
        cout << "=========================================================" << endl;
        cout << "Task 1: Distributed Mean and Standard Deviation" << endl;
        cout << "Array size: " << N << " | Processes: " << size << endl;
        cout << "=========================================================" << endl;
    }

    // Передаем sendcounts всем процессам (Scatterv требует их на root, но процессам нужно знать recvcount)
    int local_n;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Локальный буфер для части массива
    vector<int> local_data(local_n);

    // Распределяем массив между процессами
    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_INT,
                 local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Начало замера времени
    double start_time = MPI_Wtime();

    // 1. Вычисляем локальную сумму
    long long local_sum = 0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_data[i];
    }

    // 2. Вычисляем локальную сумму квадратов
    long long local_sq_sum = 0;
    for (int i = 0; i < local_n; i++) {
        local_sq_sum += static_cast<long long>(local_data[i]) * local_data[i];
    }

    // Собираем результаты на процессе 0
    long long global_sum = 0;
    long long global_sq_sum = 0;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq_sum, &global_sq_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Конец замера времени
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double mean = static_cast<double>(global_sum) / N;
        double variance = (static_cast<double>(global_sq_sum) / N) - (mean * mean);
        double std_dev = sqrt(variance);

        cout << fixed << setprecision(4);
        cout << "Global Mean:               " << mean << endl;
        cout << "Global Standard Deviation: " << std_dev << endl;
        cout << "Execution time:            " << end_time - start_time << " seconds" << endl;
        cout << "=========================================================" << endl;
    }

    MPI_Finalize();
    return 0;
}
