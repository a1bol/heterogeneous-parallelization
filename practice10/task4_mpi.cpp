#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <ctime>

// Функция для локальных вычислений (сумма)
// Function for local computation (sum)
double compute_local_sum(const std::vector<double>& data) {
    double sum = 0.0;
    for (double val : data) {
        sum += val;
    }
    return sum;
}

int main(int argc, char** argv) {
    // Инициализация MPI
    // MPI Initialization
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Общее число процессов (Total processes)

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Ранг текущего процесса (Current rank)

    // Общий размер массива данных (например, 100 миллионов)
    // Total data array size (e.g., 100 million)
    long long total_elements = 100000000;
    
    // Количество элементов на процесс
    // Elements per process
    long long elements_per_proc = total_elements / world_size;

    // Корректировка для последнего процесса (если не делится нацело)
    // Adjustment for the last process (if remainder exists)
    if (world_rank == world_size - 1) {
        elements_per_proc += total_elements % world_size;
    }

    if (world_rank == 0) {
        std::cout << "MPI Program Started with " << world_size << " processes." << std::endl;
        std::cout << "Total elements: " << total_elements << std::endl;
    }

    // Синхронизация перед началом таймера
    // Barrier before timer start
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Генерация локальных данных (симуляция)
    // Local data generation (simulation)
    // Чтобы не тратить память на хранение, можно генерировать и сразу суммировать,
    // но по заданию "обработка массива", создадим вектор.
    std::vector<double> local_data(elements_per_proc);
    
    // Простая инициализация
    // Simple initialization
    for (long long i = 0; i < elements_per_proc; ++i) {
        local_data[i] = 1.0; // Просто единицы для проверки корректности
    }

    // Локальные вычисления
    // Local computation
    double local_sum = compute_local_sum(local_data);

    // Сбор глобальной суммы
    // Gather global sum
    double global_sum = 0.0;
    
    // Используем MPI_Reduce для суммирования local_sum со всех процессов в global_sum на root (rank 0)
    // Using MPI_Reduce to sum local_sum from all processes into global_sum at root (rank 0)
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Синхронизация конца
    // End synchronization
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        std::cout << "Global Sum: " << global_sum << std::endl;
        std::cout << "Execution Time: " << (end_time - start_time) << " s" << std::endl;
        
        // Анализ масштабируемости (вывод)
        // Scalability analysis (output)
        std::cout << "Note: To test strong scaling, run with different number of processes (1, 2, 4, 8) keeping total_elements constant." << std::endl;
        std::cout << "Note: To test weak scaling, increase total_elements proportional to processes." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
