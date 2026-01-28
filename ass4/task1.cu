#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== Assignment 4 - Task 1: Array Sum Reduction (CPU vs GPU) =====
// Цель: Сравнить производительность вычисления суммы элементов массива
// на CPU (последовательно) и GPU (параллельно) с использованием глобальной памяти

// ===== CPU ВЕРСИЯ: Последовательная редукция =====

long long cpuReduction(int* arr, int n) {
    // Простая последовательная сумма
    // Алгоритм: O(n) временная сложность
    // Проходим по массиву один раз и накапливаем сумму
    
    long long sum = 0;
    
    // Каждая итерация: чтение из памяти + сложение
    for (int i = 0; i < n; ++i) {
        sum += arr[i];
    }
    
    return sum;
}

// ===== GPU ВЕРСИЯ: Параллельная редукция с глобальной памятью =====

// Kernel 1: Каждый поток суммирует свою часть данных
__global__ void parallelReduction(int* input, long long* partial_sums, int n) {
    // Вычисляем глобальный ID потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Каждый поток обрабатывает несколько элементов с шагом (stride)
    // Stride = общее количество запущенных потоков
    int stride = blockDim.x * gridDim.x;
    
    // Локальная переменная для накопления частичной суммы
    // Это находится в регистре - самая быстрая память!
    long long local_sum = 0;
    
    // Каждый поток суммирует элементы с индексами: tid, tid+stride, tid+2*stride, ...
    // Пример для 4 потоков и массива из 10 элементов:
    // Поток 0: элементы 0, 4, 8
    // Поток 1: элементы 1, 5, 9
    // Поток 2: элементы 2, 6
    // Поток 3: элементы 3, 7
    for (int i = tid; i < n; i += stride) {
        local_sum += input[i];
    }
    
    // Каждый поток записывает свою частичную сумму в глобальную память
    // ПРИМЕЧАНИЕ: Используем глобальную память как требует задание
    // (это не самый оптимальный способ - shared memory была бы быстрее)
    partial_sums[tid] = local_sum;
}

// Kernel 2: Финальная редукция частичных сумм
// Этот kernel суммирует все частичные суммы от первого kernel
__global__ void finalReduction(long long* partial_sums, long long* result, int num_threads) {
    // Используем только один блок для финальной редукции
    int tid = threadIdx.x;
    
    // Каждый поток суммирует свою часть частичных сумм
    long long local_sum = 0;
    
    for (int i = tid; i < num_threads; i += blockDim.x) {
        local_sum += partial_sums[i];
    }
    
    // Атомарное добавление к финальному результату
    // atomicAdd гарантирует корректность при параллельной записи
    // Работает медленнее обычной записи, но необходимо для корректности
    atomicAdd((unsigned long long*)result, (unsigned long long)local_sum);
}

// ===== Альтернативная GPU версия с одним ядром =====
// Более простая, но менее эффективная реализация

__global__ void simpleGpuReduction(int* input, long long* result, int n) {
    // Каждый поток обрабатывает элементы с шагом
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Локальное накопление
    long long local_sum = 0;
    for (int i = tid; i < n; i += stride) {
        local_sum += input[i];
    }
    
    // Атомарное добавление в глобальный результат
    // ПРОБЛЕМА: Все потоки конкурируют за доступ к одной ячейке памяти
    // Это создает сериализацию и снижает производительность
    atomicAdd((unsigned long long*)result, (unsigned long long)local_sum);
}

int main() {
    srand(time(0));
    
    // Размер массива: 100,000 элементов (как требуется в задании)
    const int n = 100000;
    size_t size = n * sizeof(int);
    
    cout << "=== Assignment 4 - Task 1: Array Sum Reduction ===" << endl;
    cout << "Comparing CPU (Sequential) vs GPU (Parallel) implementations" << endl;
    cout << "Array size: " << n << " elements\\n" << endl;
    
    // Выделяем память на хосте
    int* h_data = new int[n];
    
    // Инициализируем массив случайными числами
    cout << "Initializing array with random values..." << endl;
    for (int i = 0; i < n; ++i) {
        h_data[i] = rand() % 100; // Числа от 0 до 99
    }
    
    // ===== CPU РЕДУКЦИЯ =====
    
    cout << "\\n========================================" << endl;
    cout << "CPU SEQUENTIAL REDUCTION" << endl;
    cout << "========================================" << endl;
    cout << "Algorithm: Simple for-loop accumulation" << endl;
    cout << "Complexity: O(n)" << endl;
    
    auto start = high_resolution_clock::now();
    
    long long cpu_sum = cpuReduction(h_data, n);
    
    auto end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(end - start);
    
    cout << "\\nSum: " << cpu_sum << endl;
    cout << "Execution time: " << cpu_duration.count() << " μs" << endl;
    
    // ===== GPU РЕДУКЦИЯ - Метод 1: Двухэтапная редукция =====
    
    cout << "\\n========================================" << endl;
    cout << "GPU PARALLEL REDUCTION - Method 1" << endl;
    cout << "========================================" << endl;
    cout << "Algorithm: Two-stage reduction" << endl;
    cout << "Stage 1: Each thread computes partial sum" << endl;
    cout << "Stage 2: Final reduction of partial sums" << endl;
    
    // Выделяем память на GPU
    int* d_data;
    long long* d_partial_sums;
    long long* d_result;
    
    cudaMalloc(&d_data, size);
    
    // Конфигурация запуска
    int threadsPerBlock = 256;
    int blocksPerGrid = 128; // Меньше блоков для balance между параллелизмом и overhead
    int total_threads = threadsPerBlock * blocksPerGrid;
    
    cudaMalloc(&d_partial_sums, total_threads * sizeof(long long));
    cudaMalloc(&d_result, sizeof(long long));
    
    // Копируем данные на GPU
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    cout << "\\nGPU Configuration:" << endl;
    cout << "Threads per block: " << threadsPerBlock << endl;
    cout << "Blocks per grid: " << blocksPerGrid << endl;
    cout << "Total threads: " << total_threads << endl;
    
    // Обнуляем результат
    cudaMemset(d_result, 0, sizeof(long long));
    cudaMemset(d_partial_sums, 0, total_threads * sizeof(long long));
    
    start = high_resolution_clock::now();
    
    // Stage 1: Параллельная редукция с частичными суммами
    parallelReduction<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_partial_sums, n);
    
    // Stage 2: Финальная редукция
    finalReduction<<<1, threadsPerBlock>>>(d_partial_sums, d_result, total_threads);
    
    // Ждем завершения
    cudaDeviceSynchronize();
    
    end = high_resolution_clock::now();
    auto gpu_duration_v1 = duration_cast<microseconds>(end - start);
    
    // Копируем результат обратно
    long long gpu_sum_v1;
    cudaMemcpy(&gpu_sum_v1, d_result, sizeof(long long), cudaMemcpyDeviceToHost);
    
    cout << "\\nSum: " << gpu_sum_v1 << endl;
    cout << "Execution time: " << gpu_duration_v1.count() << " μs" << endl;
    cout << "Result matches CPU: " << (gpu_sum_v1 == cpu_sum ? "✓ Yes" : "✗ No") << endl;
    
    // ===== GPU РЕДУКЦИЯ - Метод 2: Простая одноэтапная редукция =====
    
    cout << "\\n========================================" << endl;
    cout << "GPU PARALLEL REDUCTION - Method 2" << endl;
    cout << "========================================" << endl;
    cout << "Algorithm: Single-stage reduction with atomicAdd" << endl;
    cout << "Simpler but potentially slower due to atomic contention" << endl;
    
    // Обнуляем результат
    cudaMemset(d_result, 0, sizeof(long long));
    
    start = high_resolution_clock::now();
    
    // Одно ядро с атомарными операциями
    simpleGpuReduction<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_result, n);
    
    cudaDeviceSynchronize();
    
    end = high_resolution_clock::now();
    auto gpu_duration_v2 = duration_cast<microseconds>(end - start);
    
    // Копируем результат
    long long gpu_sum_v2;
    cudaMemcpy(&gpu_sum_v2, d_result, sizeof(long long), cudaMemcpyDeviceToHost);
    
    cout << "\\nSum: " << gpu_sum_v2 << endl;
    cout << "Execution time: " << gpu_duration_v2.count() << " μs" << endl;
    cout << "Result matches CPU: " << (gpu_sum_v2 == cpu_sum ? "✓ Yes" : "✗ No") << endl;
    
    // ===== СРАВНИТЕЛЬНЫЙ АНАЛИЗ =====
    
    cout << "\\n========================================" << endl;
    cout << "PERFORMANCE COMPARISON" << endl;
    cout << "========================================\\n" << endl;
    
    cout << "Method              | Time (μs) | Speedup vs CPU | Correctness" << endl;
    cout << "--------------------+-----------+----------------+------------" << endl;
    cout << "CPU Sequential      |   " << cpu_duration.count() << "     |     1.00x      |     ✓" << endl;
    cout << "GPU Two-Stage       |   " << gpu_duration_v1.count() << "      |     " 
         << (double)cpu_duration.count() / gpu_duration_v1.count() << "x      |     "
         << (gpu_sum_v1 == cpu_sum ? "✓" : "✗") << endl;
    cout << "GPU Single-Stage    |   " << gpu_duration_v2.count() << "      |     " 
         << (double)cpu_duration.count() / gpu_duration_v2.count() << "x      |     "
         << (gpu_sum_v2 == cpu_sum ? "✓" : "✗") << endl;
    
    // Определяем победителя
    long long best_gpu_time = min(gpu_duration_v1.count(), gpu_duration_v2.count());
    double best_speedup = (double)cpu_duration.count() / best_gpu_time;
    
    cout << "\\n🏆 BEST GPU METHOD: " << (gpu_duration_v1.count() < gpu_duration_v2.count() ? 
         "Two-Stage Reduction" : "Single-Stage Reduction") << endl;
    cout << "Best GPU speedup: " << best_speedup << "x" << endl;
    
    if (best_speedup > 1.0) {
        cout << "\\n✓ GPU is FASTER than CPU!" << endl;
    } else {
        cout << "\\n⚠ CPU is faster (GPU overhead > benefit for this size)" << endl;
        cout << "Note: GPU shows better speedup with larger datasets" << endl;
    }
    
    // ===== АНАЛИЗ И ВЫВОДЫ =====
    
    cout << "\\n========================================" << endl;
    cout << "DETAILED ANALYSIS" << endl;
    cout << "========================================" << endl;
    
    cout << "\\n1. WHY GPU MAY BE SLOWER FOR SMALL DATASETS:" << endl;
    cout << "   - Memory transfer overhead (CPU ↔ GPU)" << endl;
    cout << "   - Kernel launch overhead" << endl;
    cout << "   - GPU underutilized with only " << n << " elements" << endl;
    cout << "   - Modern CPUs are very fast for sequential tasks" << endl;
    
    cout << "\\n2. WHEN GPU EXCELS:" << endl;
    cout << "   - Large datasets (millions+ elements)" << endl;
    cout << "   - Complex operations per element" << endl;
    cout << "   - When data is already on GPU" << endl;
    cout << "   - Multiple reductions in pipeline" << endl;
    
    cout << "\\n3. OPTIMIZATION OPPORTUNITIES:" << endl;
    cout << "   - Use shared memory instead of global (faster)" << endl;
    cout << "   - Reduce atomic operations contention" << endl;
    cout << "   - Warp-level primitives (__shfl_down_sync)" << endl;
    cout << "   - Asynchronous memory transfers" << endl;
    
    cout << "\\n4. GLOBAL MEMORY CHARACTERISTICS:" << endl;
    cout << "   - Latency: 400-800 cycles" << endl;
    cout << "   - Bandwidth: ~100-900 GB/s (GPU dependent)" << endl;
    cout << "   - Atomic operations can serialize" << endl;
    cout << "   - Coalesced access critical for performance" << endl;
    
    // Освобождаем память
    cudaFree(d_data);
    cudaFree(d_partial_sums);
    cudaFree(d_result);
    delete[] h_data;
    
    cout << "\\n========================================" << endl;
    cout << "Task completed successfully!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
