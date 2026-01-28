#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== Assignment 3 - Task 1: Global vs Shared Memory Comparison =====
// Цель: Сравнить производительность поэлементного умножения массива
// с использованием глобальной и разделяемой памяти

// ===== ВАРИАНТ 1: Использование ТОЛЬКО глобальной памяти =====

__global__ void multiplyGlobalMemory(int* input, int* output, int multiplier, int n) {
    // Вычисляем глобальный индекс потока
    // blockIdx.x - индекс блока, threadIdx.x - индекс потока в блоке
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверяем, не выходит ли индекс за границы массива
    if (idx < n) {
        // ПРОБЛЕМА: Каждое чтение и запись идет в глобальную память
        // Глобальная память - самая медленная память в GPU
        // Задержка доступа: ~400-800 циклов
        output[idx] = input[idx] * multiplier;
    }
}

// ===== ВАРИАНТ 2: Использование разделяемой (shared) памяти =====

__global__ void multiplySharedMemory(int* input, int* output, int multiplier, int n) {
    // Разделяемая память: быстрая память, разделяемая между потоками блока
    // Выделяется динамически при запуске kernel (см. третий параметр <<<>>>)
    extern __shared__ int shared_data[];
    
    // Локальный индекс потока в блоке (0 до blockDim.x-1)
    int tid = threadIdx.x;
    
    // Глобальный индекс в массиве
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Шаг 1: Загружаем данные из глобальной памяти в разделяемую
    // ПРЕИМУЩЕСТВО: Последующие операции будут работать с быстрой памятью
    // Задержка доступа shared memory: ~5 циклов (в 100 раз быстрее!)
    if (global_idx < n) {
        shared_data[tid] = input[global_idx];
    }
    
    // Синхронизация: ждем, пока все потоки блока загрузят данные
    // Критически важно! Без этого некоторые потоки могут читать неинициализированные данные
    __syncthreads();
    
    // Шаг 2: Выполняем операцию в разделяемой памяти
    // Здесь мы могли бы делать более сложные операции с соседними элементами
    // Для демонстрации делаем простое умножение
    if (global_idx < n) {
        shared_data[tid] = shared_data[tid] * multiplier;
    }
    
    // Снова синхронизация перед записью обратно
    __syncthreads();
    
    // Шаг 3: Записываем результат обратно в глобальную память
    if (global_idx < n) {
        output[global_idx] = shared_data[tid];
    }
}

int main() {
    srand(time(0));
    
    // Размер массива: 1,000,000 элементов как требуется в задании
    const int n = 1000000;
    const int multiplier = 5; // Множитель для демонстрации
    
    // Размер данных в байтах
    size_t size = n * sizeof(int);
    
    cout << "=== Assignment 3 - Task 1: Global vs Shared Memory ===" << endl;
    cout << "Array size: " << n << " elements" << endl;
    cout << "Multiplier: " << multiplier << endl;
    cout << "Operation: arr[i] = arr[i] * " << multiplier << "\\n" << endl;
    
    // Выделяем память на хосте (CPU)
    int* h_input = new int[n];
    int* h_output_global = new int[n];
    int* h_output_shared = new int[n];
    
    // Инициализируем входной массив случайными числами
    cout << "Initializing data..." << endl;
    for (int i = 0; i < n; ++i) {
        h_input[i] = rand() % 1000; // Числа от 0 до 999
    }
    
    // Выделяем память на устройстве (GPU)
    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Копируем входные данные с CPU на GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Конфигурация запуска kernel
    // 256 потоков на блок - стандартный выбор для хорошей производительности
    // Количество блоков = ceil(n / threadsPerBlock)
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    cout << "\\nGPU Configuration:" << endl;
    cout << "Threads per block: " << threadsPerBlock << endl;
    cout << "Blocks per grid: " << blocksPerGrid << endl;
    cout << "Total threads: " << threadsPerBlock * blocksPerGrid << "\\n" << endl;
    
    // ===== ТЕСТ 1: Глобальная память =====
    
    cout << "========================================" << endl;
    cout << "VERSION 1: GLOBAL MEMORY ONLY" << endl;
    cout << "========================================" << endl;
    
    // Очищаем выходной массив на GPU
    cudaMemset(d_output, 0, size);
    
    // Запускаем kernel и измеряем время
    auto start = high_resolution_clock::now();
    
    multiplyGlobalMemory<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, multiplier, n);
    
    // Ждем завершения всех операций на GPU
    cudaDeviceSynchronize();
    
    auto end = high_resolution_clock::now();
    auto global_duration = duration_cast<microseconds>(end - start);
    
    // Копируем результат обратно на CPU
    cudaMemcpy(h_output_global, d_output, size, cudaMemcpyDeviceToHost);
    
    cout << "Execution time: " << global_duration.count() << " μs" << endl;
    
    // Проверяем корректность первых элементов
    bool global_correct = true;
    for (int i = 0; i < min(10, n); ++i) {
        if (h_output_global[i] != h_input[i] * multiplier) {
            global_correct = false;
            break;
        }
    }
    cout << "Result correctness: " << (global_correct ? "✓ Correct" : "✗ Error") << endl;
    
    // ===== ТЕСТ 2: Разделяемая память =====
    
    cout << "\\n========================================" << endl;
    cout << "VERSION 2: SHARED MEMORY" << endl;
    cout << "========================================" << endl;
    
    // Очищаем выходной массив на GPU
    cudaMemset(d_output, 0, size);
    
    // Вычисляем размер разделяемой памяти
    // Каждый блок обрабатывает threadsPerBlock элементов
    size_t shared_mem_size = threadsPerBlock * sizeof(int);
    
    cout << "Shared memory per block: " << shared_mem_size << " bytes" << endl;
    
    // Запускаем kernel с использованием shared memory
    // Третий параметр (shared_mem_size) - размер динамической shared memory
    start = high_resolution_clock::now();
    
    multiplySharedMemory<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_input, d_output, multiplier, n);
    
    cudaDeviceSynchronize();
    
    end = high_resolution_clock::now();
    auto shared_duration = duration_cast<microseconds>(end - start);
    
    // Копируем результат обратно на CPU
    cudaMemcpy(h_output_shared, d_output, size, cudaMemcpyDeviceToHost);
    
    cout << "Execution time: " << shared_duration.count() << " μs" << endl;
    
    // Проверяем корректность
    bool shared_correct = true;
    for (int i = 0; i < min(10, n); ++i) {
        if (h_output_shared[i] != h_input[i] * multiplier) {
            shared_correct = false;
            break;
        }
    }
    cout << "Result correctness: " << (shared_correct ? "✓ Correct" : "✗ Error") << endl;
    
    // ===== СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ =====
    
    cout << "\\n========================================" << endl;
    cout << "PERFORMANCE COMPARISON" << endl;
    cout << "========================================" << endl;
    
    cout << "Global memory time: " << global_duration.count() << " μs" << endl;
    cout << "Shared memory time: " << shared_duration.count() << " μs" << endl;
    
    // Вычисляем ускорение (speedup)
    double speedup = (double)global_duration.count() / shared_duration.count();
    cout << "\\nSpeedup (Shared vs Global): " << speedup << "x" << endl;
    
    if (speedup > 1.0) {
        cout << "Shared memory is " << speedup << "x FASTER!" << endl;
    } else if (speedup < 1.0) {
        cout << "Global memory is faster for this simple operation" << endl;
        cout << "(Shared memory overhead > benefit for element-wise ops)" << endl;
    } else {
        cout << "Performance is similar" << endl;
    }
    
    // ===== ОБЪЯСНЕНИЕ РЕЗУЛЬТАТОВ =====
    
    cout << "\\n========================================" << endl;
    cout << "ANALYSIS" << endl;
    cout << "========================================" << endl;
    cout << "1. For simple element-wise operations, global memory can be competitive" << endl;
    cout << "2. Shared memory shows advantages when:" << endl;
    cout << "   - Accessing same data multiple times" << endl;
    cout << "   - Sharing data between threads in a block" << endl;
    cout << "   - Complex computations on local data" << endl;
    cout << "3. Memory hierarchy (fastest to slowest):" << endl;
    cout << "   Registers > Shared Memory > L1/L2 Cache > Global Memory" << endl;
    cout << "4. Latency comparison:" << endl;
    cout << "   Shared: ~5 cycles | Global: ~400-800 cycles" << endl;
    
    // Освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output_global;
    delete[] h_output_shared;
    
    return 0;
}
