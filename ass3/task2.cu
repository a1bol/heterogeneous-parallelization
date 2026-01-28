#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== Assignment 3 - Task 2: Block Size Impact on Performance =====
// Цель: Исследовать влияние размера блока потоков на производительность
// Операция: Поэлементное сложение двух массивов (C[i] = A[i] + B[i])

// CUDA kernel для поэлементного сложения массивов
// Один из самых простых GPU алгоритмов, но размер блока сильно влияет на производительность
__global__ void vectorAddition(int* a, int* b, int* c, int n) {
    // Вычисляем глобальный индекс потока
    // Каждый поток обрабатывает ровно один элемент массива
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверяем границы массива
    // Критически важно! Без этой проверки будет обращение к невыделенной памяти
    if (idx < n) {
        // Простое поэлементное сложение
        // GPU одновременно выполняет это для тысяч элементов параллельно
        c[idx] = a[idx] + b[idx];
    }
}

// Функция для проверки корректности результата
bool verifyResult(int* a, int* b, int* c, int n) {
    // Проверяем первые 100 элементов (для экономии времени)
    int check_count = min(100, n);
    for (int i = 0; i < check_count; ++i) {
        if (c[i] != a[i] + b[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    srand(time(0));
    
    // Размер массивов (достаточно большой для проявления различий)
    const int n = 10000000; // 10 миллионов элементов
    size_t size = n * sizeof(int);
    
    cout << "=== Assignment 3 - Task 2: Block Size Impact ===" << endl;
    cout << "Array size: " << n << " elements" << endl;
    cout << "Operation: C[i] = A[i] + B[i]\\n" << endl;
    
    // Выделяем память на хосте (CPU)
    int* h_a = new int[n];
    int* h_b = new int[n];
    int* h_c = new int[n];
    
    // Инициализируем входные массивы случайными числами
    cout << "Initializing input arrays..." << endl;
    for (int i = 0; i < n; ++i) {
        h_a[i] = rand() % 1000;
        h_b[i] = rand() % 1000;
    }
    
    // Выделяем память на устройстве (GPU)
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Копируем входные данные с CPU на GPU
    // Это один из узких мест производительности GPU-вычислений
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Массив различных размеров блоков для тестирования
    // ВАЖНО: Размер блока должен быть кратен размеру warp (32 потока)
    // Максимальный размер блока зависит от архитектуры GPU (обычно 1024)
    int blockSizes[] = {64, 256, 512, 1024};
    int numTests = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    cout << "\\nTesting " << numTests << " different block sizes..." << endl;
    cout << "========================================\\n" << endl;
    
    // Массивы для хранения результатов
    long long times[4];
    
    // Тестируем каждый размер блока
    for (int i = 0; i < numTests; ++i) {
        int threadsPerBlock = blockSizes[i];
        
        // Вычисляем количество блоков
        // Формула: блоки = ceil(n / threadsPerBlock)
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        
        cout << "----------------------------------------" << endl;
        cout << "TEST " << (i + 1) << ": Block size = " << threadsPerBlock << " threads" << endl;
        cout << "----------------------------------------" << endl;
        cout << "Blocks per grid: " << blocksPerGrid << endl;
        cout << "Total threads launched: " << threadsPerBlock * blocksPerGrid << endl;
        
        // Очищаем выходной массив
        cudaMemset(d_c, 0, size);
        
        // Измеряем время выполнения kernel
        // ВАЖНО: Запускаем несколько раз для более точного измерения
        const int iterations = 10;
        
        auto start = high_resolution_clock::now();
        
        for (int iter = 0; iter < iterations; ++iter) {
            // Запуск CUDA kernel
            // Синтаксис: kernel<<<blocks, threads>>>(parameters)
            vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
            
            // Ждем завершения операций на GPU
            cudaDeviceSynchronize();
        }
        
        auto end = high_resolution_clock::now();
        
        // Вычисляем среднее время за одну итерацию
        auto total_duration = duration_cast<microseconds>(end - start);
        long long avg_time = total_duration.count() / iterations;
        times[i] = avg_time;
        
        // Копируем результат обратно на CPU для проверки
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
        
        // Проверяем корректность результата
        bool correct = verifyResult(h_a, h_b, h_c, n);
        
        cout << "Average execution time: " << avg_time << " μs" << endl;
        cout << "Result correctness: " << (correct ? "✓ Correct" : "✗ Error") << endl;
        
        // Вычисляем пропускную способность (elements per second)
        double throughput = (double)n / avg_time; // миллионы элементов/секунду
        cout << "Throughput: " << throughput << " M elements/sec" << endl;
        
        cout << endl;
    }
    
    // ===== АНАЛИЗ РЕЗУЛЬТАТОВ =====
    
    cout << "========================================" << endl;
    cout << "PERFORMANCE ANALYSIS" << endl;
    cout << "========================================\\n" << endl;
    
    // Выводим сводную таблицу
    cout << "Block Size | Time (μs) | Relative Performance" << endl;
    cout << "-----------+-----------+---------------------" << endl;
    
    // Находим лучшее время для сравнения
    long long best_time = times[0];
    int best_idx = 0;
    for (int i = 1; i < numTests; ++i) {
        if (times[i] < best_time) {
            best_time = times[i];
            best_idx = i;
        }
    }
    
    // Выводим результаты относительно лучшего
    for (int i = 0; i < numTests; ++i) {
        double relative = (double)times[i] / best_time;
        cout << "   " << blockSizes[i];
        
        // Форматирование для выравнивания
        if (blockSizes[i] < 1000) cout << "    ";
        else cout << "   ";
        
        cout << " | " << times[i];
        
        // Выравнивание
        if (times[i] < 10000) cout << "     ";
        else if (times[i] < 100000) cout << "    ";
        else cout << "   ";
        
        cout << " | " << relative << "x";
        
        if (i == best_idx) {
            cout << " ⭐ BEST";
        }
        cout << endl;
    }
    
    // ===== ВЫВОДЫ И ОБЪЯСНЕНИЯ =====
    
    cout << "\\n========================================" << endl;
    cout << "KEY INSIGHTS" << endl;
    cout << "========================================" << endl;
    
    cout << "\\n1. OPTIMAL BLOCK SIZE:" << endl;
    cout << "   Best performance: " << blockSizes[best_idx] << " threads/block" << endl;
    cout << "   " << blockSizes[best_idx] << " threads is often optimal for many kernels" << endl;
    
    cout << "\\n2. WHY BLOCK SIZE MATTERS:" << endl;
    cout << "   - Too small: Underutilizes GPU cores (low occupancy)" << endl;
    cout << "   - Too large: May exceed resource limits (registers, shared mem)" << endl;
    cout << "   - Optimal: Balances occupancy and resource usage" << endl;
    
    cout << "\\n3. WARP CONSIDERATIONS:" << endl;
    cout << "   - GPU executes threads in groups of 32 (warps)" << endl;
    cout << "   - Block size should be multiple of 32" << endl;
    cout << "   - All tested sizes (64, 256, 512, 1024) are warp-aligned" << endl;
    
    cout << "\\n4. ARCHITECTURE DEPENDENCE:" << endl;
    cout << "   - Optimal size varies by GPU architecture" << endl;
    cout << "   - Modern GPUs (Ampere, Ada) prefer 256-512" << endl;
    cout << "   - Always profile on target hardware!" << endl;
    
    cout << "\\n5. OCCUPANCY:" << endl;
    cout << "   - Occupancy = (Active warps) / (Max warps)" << endl;
    cout << "   - Higher occupancy generally means better performance" << endl;
    cout << "   - But 100% occupancy is not always necessary!" << endl;
    
    // Освобождаем память
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    cout << "\\n========================================" << endl;
    cout << "Task completed successfully!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
