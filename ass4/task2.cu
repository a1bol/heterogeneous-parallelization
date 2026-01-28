#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== Assignment 4 - Task 2: Prefix Sum (Scan) on GPU =====
// Цель: Реализовать префиксную сумму (сканирование) с использованием shared memory
// Сравнить с последовательной реализацией на CPU

// Что такое Prefix Sum (Scan)?
// Дан массив: [3, 1, 7, 0, 4, 1, 6, 3]
// Prefix sum:  [3, 4, 11, 11, 15, 16, 22, 25]
// Каждый элемент = сумма всех предыдущих + текущий

// ===== CPU ВЕРСИЯ: Последовательная префиксная сумма =====

void cpuPrefixSum(int* input, int* output, int n) {
    // Простой  алгоритм O(n)
    // Каждый элемент зависит от предыдущих - сложно параллелить!
    
    if (n > 0) {
        output[0] = input[0];
        
        // Каждая итерация зависит от предыдущей
        for (int i = 1; i < n; ++i) {
            output[i] = output[i - 1] + input[i];
        }
    }
}

// ===== GPU ВЕРСИЯ: Hillis-Steele Scan в Shared Memory =====

// Алгоритм Hillis-Steele: параллельный prefix sum
// Работает за O(log n) шагов, но требует O(n log n) операций
__global__ void prefixSumShared(int* input, int* output, int n) {
    // Разделяемая память для блока
    // Используем двойную буферизацию для избежания race conditions
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Загружаем данные в shared memory
    if (global_tid < n) {
        temp[tid] = input[global_tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    
    // Hillis-Steele алгоритм
    // На каждом шаге увеличиваем offset в 2 раза: 1, 2, 4, 8...
    // Пример для 8 элементов [3, 1, 7, 0, 4, 1, 6, 3]:
    //
    // Исходные:    [3, 1, 7, 0, 4, 1, 6, 3]
    // Offset=1:    [3, 4, 8, 7, 4, 5, 7, 9]  ← temp[i] += temp[i-1]
    // Offset=2:    [3, 4, 11,11,12,12,11,14] ← temp[i] += temp[i-2]
    // Offset=4:    [3, 4, 11,11,15,16,22,25] ← temp[i] += temp[i-4]
    
    int blockSize = blockDim.x;
    
    for (int offset = 1; offset < blockSize; offset *= 2) {
        int val = 0;
        
        // Читаем значение с учетом offset
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        
        __syncthreads(); // Критическая синхронизация!
        
        // Добавляем к текущему значению
        if (tid >= offset) {
            temp[tid] += val;
        }
        
        __syncthreads(); // Ждем завершения записи
    }
    
    // Записываем результат обратно в глобальную память
    if (global_tid < n) {
        output[global_tid] = temp[tid];
    }
}

// Kernel для объединения результатов нескольких блоков
// После сканирования каждого блока нужно добавить сумму предыдущих блоков
__global__ void addBlockSums(int* data, int* blockSums, int n, int blockSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Блок 0 уже корректен, начинаем с блока 1
    if (blockIdx.x > 0 && tid < n) {
        // Добавляем сумму всех предыдущих блоков
        data[tid] += blockSums[blockIdx.x - 1];
    }
}

// Функция для многоблочного prefix sum
void gpuPrefixSum(int* h_input, int* h_output, int n) {
    // Размер блока
    const int BLOCK_SIZE = 512; // Максимум для prefix sum в shared memory
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Выделяем память на GPU
    int *d_input, *d_output, *d_blockSums;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));
    
    // Копируем данные
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Размер shared memory на блок
    int sharedMemSize = BLOCK_SIZE * sizeof(int);
    
    // Шаг 1: Выполняем prefix sum для каждого блока независимо
    prefixSumShared<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    // Если больше одного блока, нужно объединить результаты
    if (numBlocks > 1) {
        // Шаг 2: Собираем последние элементы каждого блока (суммы блоков)
        int* h_blockSums = new int[numBlocks];
        
        for (int i = 0; i < numBlocks; ++i) {
            int lastIdx = min((i + 1) * BLOCK_SIZE - 1, n - 1);
            cudaMemcpy(&h_blockSums[i], &d_output[lastIdx], sizeof(int), cudaMemcpyDeviceToHost);
        }
        
        // Шаг 3: Вычисляем prefix sum сумм блоков (на CPU для простоты)
        for (int i = 1; i < numBlocks; ++i) {
            h_blockSums[i] += h_blockSums[i - 1];
        }
        
        cudaMemcpy(d_blockSums, h_blockSums, numBlocks * sizeof(int), cudaMemcpyHostToDevice);
        
        // Шаг 4: Добавляем суммы предыдущих блоков к каждому элементу
        addBlockSums<<<numBlocks, BLOCK_SIZE>>>(d_output, d_blockSums, n, BLOCK_SIZE);
        cudaDeviceSynchronize();
        
        delete[] h_blockSums;
    }
    
    // Копируем результат обратно
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
}

// Функция проверки корректности
bool verifyPrefixSum(int* input, int* output, int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += input[i];
        if (output[i] != sum) {
            cout << "Error at index " << i << ": expected " << sum 
                 << ", got " << output[i] << endl;
            return false;
        }
    }
    return true;
}

int main() {
    srand(time(0));
    
    // Размер массива: 1,000,000 элементов (как требуется в задании)
    const int n = 1000000;
    
    cout << "=== Assignment 4 - Task 2: Prefix Sum (Scan) ===" << endl;
    cout << "Array size: " << n << " elements" << endl;
    cout << "Algorithm: Hillis-Steele scan with shared memory\\n" << endl;
    
    // Выделяем память на хосте
    int* h_input = new int[n];
    int* h_output_cpu = new int[n];
    int* h_output_gpu = new int[n];
    
    // Инициализируем входной массив
    cout << "Initializing array with random values..." << endl;
    for (int i = 0; i < n; ++i) {
        h_input[i] = rand() % 10; // Числа от 0 до 9
    }
    
    // Показываем первые элементы для наглядности
    cout << "\\nFirst 10 input elements: ";
    for (int i = 0; i < 10; ++i) {
        cout << h_input[i] << " ";
    }
    cout << endl;
    
    // ===== CPU ВЕРСИЯ =====
    
    cout << "\\n========================================" << endl;
    cout << "CPU SEQUENTIAL PREFIX SUM" << endl;
    cout << "========================================" << endl;
    cout << "Algorithm: Simple iterative scan" << endl;
    cout << "Complexity: O(n) time, O(1) space" << endl;
    
    auto start = high_resolution_clock::now();
    
    cpuPrefixSum(h_input, h_output_cpu, n);
    
    auto end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<milliseconds>(end - start);
    
    cout << "\\nExecution time: " << cpu_duration.count() << " ms" << endl;
    
    // Показываем первые результаты
    cout << "First 10 output elements: ";
    for (int i = 0; i < 10; ++i) {
        cout << h_output_cpu[i] << " ";
    }
    cout << endl;
    cout << "Last element (total sum): " << h_output_cpu[n-1] << endl;
    
    // ===== GPU ВЕРСИЯ =====
    
    cout << "\\n========================================" << endl;
    cout << "GPU PARALLEL PREFIX SUM" << endl;
    cout << "========================================" << endl;
    cout << "Algorithm: Hillis-Steele scan" << endl;
    cout << "Complexity: O(log n) time, O(n) work" << endl;
    cout << "Optimization: Shared memory per block" << endl;
    
    start = high_resolution_clock::now();
    
    gpuPrefixSum(h_input, h_output_gpu, n);
    
    end = high_resolution_clock::now();
    auto gpu_duration = duration_cast<milliseconds>(end - start);
    
    cout << "\\nExecution time: " << gpu_duration.count() << " ms" << endl;
    
    // Показываем первые результаты
    cout << "First 10 output elements: ";
    for (int i = 0; i < 10; ++i) {
        cout << h_output_gpu[i] << " ";
    }
    cout << endl;
    cout << "Last element (total sum): " << h_output_gpu[n-1] << endl;
    
    // Проверяем корректность
    bool gpu_correct = verifyPrefixSum(h_input, h_output_gpu, n);
    cout << "Result correctness: " << (gpu_correct ? "✓ Correct" : "✗ Error") << endl;
    
    // ===== СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ =====
    
    cout << "\\n========================================" << endl;
    cout << "PERFORMANCE COMPARISON" << endl;
    cout << "========================================\\n" << endl;
    
    cout << "Method            | Time (ms) | Speedup vs CPU" << endl;
    cout << "------------------+-----------+----------------" << endl;
    cout << "CPU Sequential    |   " << cpu_duration.count() << "     |     1.00x (baseline)" << endl;
    cout << "GPU Parallel      |   " << gpu_duration.count() << "     |     " 
         << (double)cpu_duration.count() / gpu_duration.count() << "x" << endl;
    
    double speedup = (double)cpu_duration.count() / gpu_duration.count();
    
    if (speedup > 1.0) {
        cout << "\\n✓ GPU is " << speedup << "x FASTER!" << endl;
    } else {
        cout << "\\n⚠ CPU is faster for this problem size" << endl;
        cout << "Note: Prefix sum has strong sequential dependencies" << endl;
    }
    
    // ===== АНАЛИЗ И ОБЪЯСНЕНИЯ =====
    
    cout << "\\n========================================" << endl;
    cout << "ALGORITHM ANALYSIS" << endl;
    cout << "========================================" << endl;
    
    cout << "\\n1. WHY PREFIX SUM IS CHALLENGING ON GPU:" << endl;
    cout << "   - Strong sequential dependencies" << endl;
    cout << "   - Each element depends on all previous elements" << endl;
    cout << "   - Work complexity O(n log n) vs CPU O(n)" << endl;
    cout << "   - Requires careful synchronization" << endl;
    
    cout << "\\n2. HILLIS-STEELE VS BLELLOCH:" << endl;
    cout << "   Hillis-Steele (this implementation):" << endl;
    cout << "   ✓ Simpler to implement" << endl;
    cout << "   ✓ Better for small arrays" << endl;
    cout << "   ✗ O(n log n) work (inefficient)" << endl;
    cout << "   " << endl;
    cout << "   Blelloch (work-efficient):" << endl;
    cout << "   ✓ O(n) work (optimal)" << endl;
    cout << "   ✓ Better for large arrays" << endl;
    cout << "   ✗ More complex implementation" << endl;
    
    cout << "\\n3. SHARED MEMORY BENEFITS:" << endl;
    cout << "   - Fast intra-block communication" << endl;
    cout << "   - Reduces global memory accesses" << endl;
    cout << "   - Enables efficient synchronization" << endl;
    cout << "   - Critical for scan algorithms" << endl;
    
    cout << "\\n4. MULTI-BLOCK CHALLENGES:" << endl;
    cout << "   - Need to combine results across blocks" << endl;
    cout << "   - Recursive scan of block sums" << endl;
    cout << "   - Additional synchronization overhead" << endl;
    cout << "   - This implementation uses CPU for block sum scan" << endl;
    
    cout << "\\n5. APPLICATIONS OF PREFIX SUM:" << endl;
    cout << "   - Stream compaction" << endl;
    cout << "   - Radix sort" << endl;
    cout << "   - Allocation/packing algorithms" << endl;
    cout << "   - Graph algorithms (BFS, shortest paths)" << endl;
    cout << "   - String matching" << endl;
    
    cout << "\\n6. OPTIMIZATION OPPORTUNITIES:" << endl;
    cout << "   ✓ Use Blelloch algorithm for better work efficiency" << endl;
    cout << "   ✓ Implement recursive block sum scan on GPU" << endl;
    cout << "   ✓ Use warp-level primitives (__shfl_up_sync)" << endl;
    cout << "   ✓ Consider CUB library for production code" << endl;
    
    // Освобождаем память
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    
    cout << "\\n========================================" << endl;
    cout << "Task completed successfully!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
