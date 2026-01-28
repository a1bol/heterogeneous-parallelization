#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

// ===== Assignment 4 - Task 4: MPI Distributed Processing =====
// Цель: Реализовать распределенную программу с использованием MPI
// Операция: Обработка массива данных с распределением между процессами

// Функция обработки данных (вычислительно-интенсивная операция)
// Для демонстрации: возведение в степень и суммирование
void processData(int* data, int* result, int n) {
    for (int i = 0; i < n; ++i) {
        // Искусственная вычислительная нагрузка
        int value = data[i];
        result[i] = value * value + value / 2;
    }
}

// Функция для вычисления суммы массива
long long computeSum(int* data, int n) {
    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

int main(int argc, char** argv) {
    // Инициализация MPI
    // ВАЖНО: Должна быть первым вызовом MPI
    MPI_Init(&argc, &argv);
    
    int world_size;  // Общее количество процессов
    int world_rank;  // ID текущего процесса (0 до world_size-1)
    
    // Получаем информацию о процессах
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Размер глобального массива (будет разделен между процессами)
    const int TOTAL_SIZE = 10000000; // 10 миллионов элементов
    
    // Вычисляем размер данных для каждого процесса
    // Последний процесс может получить чуть больше, если не делится нацело
    int local_size = TOTAL_SIZE / world_size;
    int remainder = TOTAL_SIZE % world_size;
    
    // Процесс 0 (master) готовит и распределяет данные
    int* global_data = nullptr;
    int* global_result = nullptr;
    
    if (world_rank == 0) {
        cout << "=== Assignment 4 - Task 4: MPI Distributed Processing ===" << endl;
        cout << "Total processes: " << world_size << endl;
        cout << "Array size: " << TOTAL_SIZE << " elements" << endl;
        cout << "Elements per process: ~" << local_size << "\\n" << endl;
        
        // Выделяем память для глобальных массивов на master процессе
        global_data = new int[TOTAL_SIZE];
        global_result = new int[TOTAL_SIZE];
        
        // Инициализируем данные случайными числами
        srand(time(0));
        cout << "Master (rank 0): Initializing data..." << endl;
        for (int i = 0; i < TOTAL_SIZE; ++i) {
            global_data[i] = rand() % 1000;
        }
    }
    
    // Выделяем буферы для локальных данных каждого процесса
    int* local_data = new int[local_size + remainder];
    int* local_result = new int[local_size + remainder];
    
    // Засекаем время для распределенных вычислений
    auto start_time = high_resolution_clock::now();
    
    // ===== ШАГ 1: РАСПРЕДЕЛЕНИЕ ДАННЫХ =====
    
    if (world_rank == 0) {
        cout << "\\n========================================" << endl;
        cout << "STEP 1: SCATTERING DATA" << endl;
        cout << "========================================" << endl;
        cout << "Distributing data from master to all processes..." << endl;
    }
    
    // MPI_Scatter: Распределяет данные от процесса 0 ко всем процессам
    // Каждый процесс получает свою часть массива
    MPI_Scatter(
        global_data,        // Исходные данные (на процессе 0)
        local_size,         // Сколько элементов отправить каждому
        MPI_INT,            // Тип данных
        local_data,         // Где сохранить полученные данные
        local_size,         // Сколько элементов получим
        MPI_INT,            // Тип получаемых данных
        0,                  // Процесс-отправитель (root)
        MPI_COMM_WORLD      // Коммуникатор (группа процессов)
    );
    
    // Если есть остаток, процесс 0 обрабатывает дополнительные элементы
    int actual_local_size = local_size;
    if (world_rank == 0 && remainder > 0) {
        for (int i = 0; i < remainder; ++i) {
            local_data[local_size + i] = global_data[world_size * local_size + i];
        }
        actual_local_size += remainder;
    }
    
    if (world_rank == 0) {
        cout << "Data scattered successfully!" << endl;
    }
    
    // Барьер: Все процессы ждут друг друга перед началом обработки
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== ШАГ 2: ЛОКАЛЬНАЯ ОБРАБОТКА =====
    
    if (world_rank == 0) {
        cout << "\\n========================================" << endl;
        cout << "STEP 2: PARALLEL PROCESSING" << endl;
        cout << "========================================" << endl;
        cout << "Each process computing independently..." << endl;
    }
    
    // Каждый процесс независимо обрабатывает свою часть данных
    auto local_start = high_resolution_clock::now();
    
    processData(local_data, local_result, actual_local_size);
    
    auto local_end = high_resolution_clock::now();
    auto local_duration = duration_cast<microseconds>(local_end - local_start);
    
    // Каждый процесс выводит свое время (может быть неупорядоченно!)
    printf("Rank %d: Processed %d elements in %lld μs\\n", 
           world_rank, actual_local_size, local_duration.count());
    
    // Барьер: Ждем завершения всех процессов
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== ШАГ 3: СБОР РЕЗУЛЬТАТОВ =====
    
    if (world_rank == 0) {
        cout << "\\n========================================" << endl;
        cout << "STEP 3: GATHERING RESULTS" << endl;
        cout << "========================================" << endl;
        cout << "Collecting results from all processes..." << endl;
    }
    
    // MPI_Gather: Собирает результаты от всех процессов обратно к процессу 0
    MPI_Gather(
        local_result,       // Локальные данные для отправки
        local_size,         // Количество элементов
        MPI_INT,            // Тип данных
        global_result,      // Где сохранить собранные данные (на процессе 0)
        local_size,         // Сколько получить от каждого процесса
        MPI_INT,            // Тип данных
        0,                  // Процесс-получатель (root)
        MPI_COMM_WORLD      // Коммуникатор
    );
    
    // Процесс 0 добавляет результаты остатка
    if (world_rank == 0 && remainder > 0) {
        for (int i = 0; i < remainder; ++i) {
            global_result[world_size * local_size + i] = local_result[local_size + i];
        }
    }
    
    auto end_time = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(end_time - start_time);
    
    if (world_rank == 0) {
        cout << "Results gathered successfully!" << endl;
    }
    
    // ===== ШАГ 4: АГРЕГАЦИЯ И АНАЛИЗ (только процесс 0) =====
    
    if (world_rank == 0) {
        cout << "\\n========================================" << endl;
        cout << "STEP 4: FINAL AGGREGATION" << endl;
        cout << "========================================" << endl;
        
        // Вычисляем сумму для проверки
        long long total_sum = computeSum(global_result, TOTAL_SIZE);
        
        cout << "Total sum of results: " << total_sum << endl;
        cout << "\\nTotal distributed processing time: " << total_duration.count() << " ms" << endl;
        
        // ===== СРАВНЕНИЕ С ПОСЛЕДОВАТЕЛЬНОЙ ОБРАБОТКОЙ =====
        
        cout << "\\n========================================" << endl;
        cout << "SEQUENTIAL COMPARISON" << endl;
        cout << "========================================" << endl;
        cout << "Processing entire array on single process..." << endl;
        
        int* sequential_result = new int[TOTAL_SIZE];
        
        auto seq_start = high_resolution_clock::now();
        processData(global_data, sequential_result, TOTAL_SIZE);
        auto seq_end = high_resolution_clock::now();
        auto seq_duration = duration_cast<milliseconds>(seq_end - seq_start);
        
        cout << "Sequential processing time: " << seq_duration.count() << " ms" << endl;
        
        // Проверяем корректность результатов
        bool correct = true;
        for (int i = 0; i < min(1000, TOTAL_SIZE); ++i) {
            if (global_result[i] != sequential_result[i]) {
                correct = false;
                break;
            }
        }
        
        cout << "Result correctness: " << (correct ? "✓ Correct" : "✗ Error") << endl;
        
        // ===== АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ =====
        
        cout << "\\n========================================" << endl;
        cout << "PERFORMANCE ANALYSIS" << endl;
        cout << "========================================\\n" << endl;
        
        cout << "Processes | Distributed Time | Sequential Time | Speedup" << endl;
        cout << "----------+------------------+-----------------+---------" << endl;
        cout << "    " << world_size << "     |      " << total_duration.count() 
             << " ms      |     " << seq_duration.count() << " ms     |   " 
             << (double)seq_duration.count() / total_duration.count() << "x" << endl;
        
        double speedup = (double)seq_duration.count() / total_duration.count();
        double efficiency = speedup / world_size * 100.0;
        
        cout << "\\n📊 METRICS:" << endl;
        cout << "Speedup: " << speedup << "x" << endl;
        cout << "Efficiency: " << efficiency << "% (ideal = 100%)" << endl;
        cout << "Parallel overhead: " << (world_size * total_duration.count() - seq_duration.count()) << " ms" << endl;
        
        // ===== ОБЪЯСНЕНИЕ И ВЫВОДЫ =====
        
        cout << "\\n========================================" << endl;
        cout << "KEY INSIGHTS" << endl;
        cout << "========================================" << endl;
        
        cout << "\\n1. SCALABILITY:" << endl;
        cout << "   - " << world_size << " processes achieved " << speedup << "x speedup" << endl;
        cout << "   - Efficiency: " << efficiency << "%" << endl;
        if (efficiency > 80) {
            cout << "   ✓ Excellent scalability!" << endl;
        } else if (efficiency > 60) {
            cout << "   ✓ Good scalability" << endl;
        } else {
            cout << "   ⚠ Communication overhead is significant" << endl;
        }
        
        cout << "\\n2. WHY NOT LINEAR SPEEDUP?" << endl;
        cout << "   - Communication overhead (scatter/gather)" << endl;
        cout << "   - Synchronization points (barriers)" << endl;
        cout << "   - Load imbalance (if work varies)" << endl;
        cout << "   - Sequential portions (Amdahl's law)" << endl;
        
        cout << "\\n3. OPTIMAL NUMBER OF PROCESSES:" << endl;
        cout << "   - Depends on problem size and communication cost" << endl;
        cout << "   - More processes ≠ always faster" << endl;
        cout << "   - Sweet spot: balance computation vs communication" << endl;
        cout << "   - For this problem: test 2, 4, 8 processes" << endl;
        
        cout << "\\n4. MPI BENEFITS:" << endl;
        cout << "   ✓ Distributed memory across nodes" << endl;
        cout << "   ✓ Scales to thousands of processes" << endl;
        cout << "   ✓ Standard for HPC clusters" << endl;
        cout << "   ✓ Language-agnostic (C, C++, Fortran, Python)" << endl;
        
        cout << "\\n5. MPI CHALLENGES:" << endl;
        cout << "   ⚠ Complex programming model" << endl;
        cout << "   ⚠ Explicit data distribution" << endl;
        cout << "   ⚠ Debugging is harder" << endl;
        cout << "   ⚠ Network can be bottleneck" << endl;
        
        cout << "\\n6. WHEN TO USE MPI:" << endl;
        cout << "   ✓ Data too large for single machine" << endl;
        cout << "   ✓ Embarrassingly parallel problems" << endl;
        cout << "   ✓ HPC clusters available" << endl;
        cout << "   ✓ Long-running simulations" << endl;
        
        // Освобождаем память
        delete[] global_data;
        delete[] global_result;
        delete[] sequential_result;
        
        cout << "\\n========================================" << endl;
        cout << "Task completed successfully!" << endl;
        cout << "========================================" << endl;
    }
    
    // Освобождаем локальную память
    delete[] local_data;
    delete[] local_result;
    
    // Финализация MPI
    // ВАЖНО: Должна быть последним вызовом MPI
    MPI_Finalize();
    
    return 0;
}
