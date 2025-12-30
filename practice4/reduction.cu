#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== ВАРИАНТ 1: Редукция используя ТОЛЬКО глобальную память =====

__global__ void reductionGlobalMemory(int* input, int* output, int n) {
    // Вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Каждый поток обрабатывает stride элементов
    // Stride = общее количество потоков
    int stride = blockDim.x * gridDim.x;
    
    // Локальная переменная для накопления суммы
    int local_sum = 0;
    
    // Суммируем элементы с шагом stride
    for (int i = tid; i < n; i += stride) {
        local_sum += input[i];
    }
    
    // ПРОБЛЕМА: Запись в глобальную память медленная!
    // Каждый поток пишет свою partial sum в глобальную память
    // Это создает много конфликтов доступа
    atomicAdd(&output[0], local_sum);
}

// ===== ВАРИАНТ 2: Редукция используя глобальную + разделяемую память =====

__global__ void reductionSharedMemory(int* input, int* output, int n) {
    // Разделяемая память: быстрая, доступна всем потокам в блоке
    // Выделяется динамически при запуске ядра
    extern __shared__ int shared_data[];
    
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Шаг 1: Каждый поток суммирует свои элементы в локальную переменную
    int local_sum = 0;
    for (int i = global_tid; i < n; i += stride) {
        local_sum += input[i];
    }
    
    // Шаг 2: Записываем локальную сумму в разделяемую память
    // ПРЕИМУЩЕСТВО: Разделяемая память намного быстрее глобальной!
    shared_data[tid] = local_sum;
    __syncthreads(); // Ждем пока все потоки запишут данные
    
    // Шаг 3: Параллельная редукция в разделяемой памяти
    // Используем алгоритм "дерева": каждый уровень уменьшает количество активных потоков вдвое
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Суммируем пары элементов
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads(); // Синхронизация между итерациями
    }
    
    // Шаг 4: Первый поток блока записывает результат блока в глобальную память
    // Теперь вместо 256 записей в глобальную память - всего 1!
    if (tid == 0) {
        atomicAdd(&output[0], shared_data[0]);
    }
}

// ===== CPU версия для сравнения =====

long long cpuReduction(int* data, int n) {
    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

// ===== MAIN ФУНКЦИЯ =====

int main() {
    srand(time(0));
    
    // Тестируем на разных размерах
    int sizes[] = {10000, 100000, 1000000};
    
    cout << "=== Practice 4: Reduction Optimization ===" << endl;
    cout << "Comparing Global Memory vs Global+Shared Memory\\n" << endl;
    
    for (int size : sizes) {
        cout << "\\n========================================" << endl;
        cout << "Array size: " << size << endl;
        cout << "========================================" << endl;
        
        // Выделяем память на хосте
        int* h_data = new int[size];
        int* h_result = new int[1];
        
        // Генерируем случайные данные
        for (int i = 0; i < size; ++i) {
            h_data[i] = rand() % 100;
        }
        
        // === CPU Редукция (для проверки) ===
        auto start = high_resolution_clock::now();
        long long cpu_sum = cpuReduction(h_data, size);
        auto end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<microseconds>(end - start);
        
        cout << "\\nCPU Reduction:" << endl;
        cout << "Sum: " << cpu_sum << endl;
        cout << "Time: " << cpu_duration.count() << " μs" << endl;
        
        // Выделяем память на GPU
        int *d_data, *d_result;
        cudaMalloc(&d_data, size * sizeof(int));
        cudaMalloc(&d_result, sizeof(int));
        
        // Копируем данные на GPU
        cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
        
        // Конфигурация запуска
        int threadsPerBlock = 256;
        int blocks = min((size + threadsPerBlock - 1) / threadsPerBlock, 256);
        
        // === ВАРИАНТ 1: Только глобальная память ===
        cudaMemset(d_result, 0, sizeof(int));
        
        start = high_resolution_clock::now();
        reductionGlobalMemory<<<blocks, threadsPerBlock>>>(d_data, d_result, size);
        cudaDeviceSynchronize();
        end = high_resolution_clock::now();
        auto global_duration = duration_cast<microseconds>(end - start);
        
        cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
        
        cout << "\\nGPU Reduction (Global Memory Only):" << endl;
        cout << "Sum: " << h_result[0] << endl;
        cout << "Time: " << global_duration.count() << " μs" << endl;
        cout << "Correct: " << (h_result[0] == cpu_sum ? "✓ Yes" : "✗ No") << endl;
        
        // === ВАРИАНТ 2: Глобальная + Разделяемая память ===
        cudaMemset(d_result, 0, sizeof(int));
        
        int shared_mem_size = threadsPerBlock * sizeof(int);
        
        start = high_resolution_clock::now();
        reductionSharedMemory<<<blocks, threadsPerBlock, shared_mem_size>>>(d_data, d_result, size);
        cudaDeviceSynchronize();
        end = high_resolution_clock::now();
        auto shared_duration = duration_cast<microseconds>(end - start);
        
        cudaMemcpy(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
        
        cout << "\\nGPU Reduction (Global + Shared Memory):" << endl;
        cout << "Sum: " << h_result[0] << endl;
        cout << "Time: " << shared_duration.count() << " μs" << endl;
        cout << "Correct: " << (h_result[0] == cpu_sum ? "✓ Yes" : "✗ No") << endl;
        
        // === Сравнение ===
        double speedup = (double)global_duration.count() / shared_duration.count();
        cout << "\\n=== Performance Comparison ===" << endl;
        cout << "Speedup (Shared vs Global): " << speedup << "x" << endl;
        
        if (speedup > 1.0) {
            cout << "Shared memory is " << speedup << "x faster!" << endl;
        } else {
            cout << "Similar performance" << endl;
        }
        
        // Очистка
        cudaFree(d_data);
        cudaFree(d_result);
        delete[] h_data;
        delete[] h_result;
    }
    
    cout << "\\n========================================" << endl;
    cout << "Summary:" << endl;
    cout << "========================================" << endl;
    cout << "1. Shared memory is much faster than global memory" << endl;
    cout << "2. Reduces global memory accesses significantly" << endl;
    cout << "3. Enables efficient parallel reduction within block" << endl;
    cout << "4. Speedup increases with problem size" << endl;
    cout << "\\nMemory Hierarchy (Fastest to Slowest):" << endl;
    cout << "  Registers > Shared Memory > Global Memory" << endl;
    
    return 0;
}
