#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== Assignment 3 - Task 3: Coalesced vs Non-Coalesced Memory Access =====
// Цель: Продемонстрировать влияние паттерна доступа к глобальной памяти
// на производительность GPU программы

// ===== ВАРИАНТ 1: Коалесцированный (Coalesced) доступ к памяти =====

// Что такое коалесцированный доступ?
// Когда потоки в warp (группа из 32 потоков) обращаются к последовательным
// адресам памяти, GPU может объединить эти обращения в одну транзакцию памяти
// Это НАМНОГО быстрее, чем множество отдельных транзакций!

__global__ void coalescedAccess(int* input, int* output, int n) {
    // Вычисляем глобальный индекс
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // КОАЛЕСЦИРОВАННЫЙ доступ: stride = 1
        // Поток 0 читает элемент 0, поток 1 - элемент 1, и т.д.
        // Все потоки в warp читат последовательные адреса памяти
        // 
        // Память GPU организована в "cache lines" (обычно 32-128 байт)
        // Когда потоки warp читают последовательную память, вся cache line
        // загружается одной транзакцией - это очень эффективно!
        //
        // Визуализация для warp из 32 потоков:
        // Thread:  0  1  2  3  4  5 ... 30 31
        // Access: [0][1][2][3][4][5]...[30][31]  ← Последовательно!
        
        output[idx] = input[idx] * 2;
    }
}

// ===== ВАРИАНТ 2: Некоалесцированный (Non-Coalesced) доступ =====

__global__ void nonCoalescedAccess(int* input, int* output, int n) {
    // Вычисляем глобальный индекс
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // НЕКОАЛЕСЦИРОВАННЫЙ доступ: stride = 32
        // Поток 0 читает элемент 0, поток 1 - элемент 32, поток 2 - элемент 64...
        // Потоки в warp читают адреса с большими промежутками
        //
        // ПРОБЛЕМА: Каждый поток в warp обращается к разным cache lines!
        // GPU должен выполнить 32 отдельные транзакции памяти вместо одной
        // Это в 32 раза медленнее в худшем случае!
        //
        // Визуализация для warp из 32 потоков:
        // Thread:  0    1    2    3   ...  31
        // Access: [0] [32] [64] [96] ... [992]  ← Разбросаны!
        //
        // Stride = 32 выбран намеренно как наихудший случай
        // (размер warp = 32, поэтому каждый поток попадает в свою cache line)
        
        int strided_idx = idx * 32;
        
        if (strided_idx < n) {
            output[strided_idx] = input[strided_idx] * 2;
        }
    }
}

// ===== ВАРИАНТ 3: Для сравнения - еще один некоалесцированный паттерн =====

__global__ void reverseAccess(int* input, int* output, int n) {
    // Обратный доступ к памяти
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Обращаемся к памяти в обратном порядке
        // Менее плохо чем stride=32, но все еще некоалесцированно
        // Потоки warp обращаются к памяти в обратном порядке блоками
        
        // Визуализация:
        // Thread:  0  1  2  3 ...  31
        // Access: [31][30][29][28]...[0]  ← Обратный порядок
        
        int reverse_idx = n - 1 - idx;
        output[reverse_idx] = input[reverse_idx] * 2;
    }
}

// Функция проверки корректности результата
bool verifyResult(int* input, int* output, int n, int stride) {
    for (int i = 0; i < n; i += stride) {
        if (output[i] != input[i] * 2) {
            cout << "Error at index " << i << ": expected " << (input[i] * 2) 
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
    size_t size = n * sizeof(int);
    
    cout << "=== Assignment 3 - Task 3: Memory Access Patterns ===" << endl;
    cout << "Array size: " << n << " elements" << endl;
    cout << "Operation: output[i] = input[i] * 2\\n" << endl;
    
    // Выделяем память на хосте
    int* h_input = new int[n];
    int* h_output = new int[n];
    
    // Инициализируем входной массив
    cout << "Initializing data..." << endl;
    for (int i = 0; i < n; ++i) {
        h_input[i] = rand() % 1000;
    }
    
    // Выделяем память на устройстве
    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    // Копируем данные на GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Конфигурация запуска
    int threadsPerBlock = 256; // Оптимальный размер блока
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    cout << "GPU Configuration:" << endl;
    cout << "Threads per block: " << threadsPerBlock << endl;
    cout << "Blocks per grid: " << blocksPerGrid << endl;
    cout << "Total threads: " << threadsPerBlock * blocksPerGrid << "\\n" << endl;
    
    // ===== ТЕСТ 1: Коалесцированный доступ =====
    
    cout << "========================================" << endl;
    cout << "TEST 1: COALESCED ACCESS (stride = 1)" << endl;
    cout << "========================================" << endl;
    cout << "Pattern: Sequential memory access" << endl;
    cout << "Thread i accesses element i" << endl;
    
    // Очищаем выходной массив
    cudaMemset(d_output, 0, size);
    
    // Запускаем несколько раз для более точного измерения
    const int iterations = 20;
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        coalescedAccess<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
    
    auto end = high_resolution_clock::now();
    auto coalesced_duration = duration_cast<microseconds>(end - start);
    long long avg_coalesced = coalesced_duration.count() / iterations;
    
    // Копируем результат для проверки
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    bool coalesced_correct = verifyResult(h_input, h_output, n, 1);
    
    cout << "\\nAverage execution time: " << avg_coalesced << " μs" << endl;
    cout << "Result correctness: " << (coalesced_correct ? "✓ Correct" : "✗ Error") << endl;
    
    // Вычисляем пропускную способность памяти
    // Формула: (bytes read + bytes written) / time
    double bandwidth_coalesced = (2.0 * size) / (avg_coalesced * 1e-6) / 1e9; // GB/s
    cout << "Memory bandwidth: " << bandwidth_coalesced << " GB/s" << endl;
    
    // ===== ТЕСТ 2: Некоалесцированный доступ (stride = 32) =====
    
    cout << "\\n========================================" << endl;
    cout << "TEST 2: NON-COALESCED ACCESS (stride = 32)" << endl;
    cout << "========================================" << endl;
    cout << "Pattern: Strided memory access" << endl;
    cout << "Thread i accesses element i*32" << endl;
    cout << "Worst case: each thread in warp hits different cache line" << endl;
    
    // Очищаем массивы
    cudaMemset(d_output, 0, size);
    memset(h_output, 0, size);
    
    // Специальная конфигурация для strided доступа
    // Нужно меньше потоков, так как каждый поток обрабатывает элемент с stride=32
    int effective_elements = n / 32;
    int blocks_strided = (effective_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        nonCoalescedAccess<<<blocks_strided, threadsPerBlock>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
    
    end = high_resolution_clock::now();
    auto strided_duration = duration_cast<microseconds>(end - start);
    long long avg_strided = strided_duration.count() / iterations;
    
    // Копируем и проверяем результат
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    bool strided_correct = verifyResult(h_input, h_output, n, 32);
    
    cout << "\\nAverage execution time: " << avg_strided << " μs" << endl;
    cout << "Result correctness: " << (strided_correct ? "✓ Correct" : "✗ Error") << endl;
    
    double bandwidth_strided = (2.0 * size / 32) / (avg_strided * 1e-6) / 1e9; // GB/s
    cout << "Memory bandwidth: " << bandwidth_strided << " GB/s" << endl;
    
    // ===== ТЕСТ 3: Обратный доступ =====
    
    cout << "\\n========================================" << endl;
    cout << "TEST 3: REVERSE ACCESS" << endl;
    cout << "========================================" << endl;
    cout << "Pattern: Reverse order access" << endl;
    cout << "Thread i accesses element (n-1-i)" << endl;
    
    cudaMemset(d_output, 0, size);
    memset(h_output, 0, size);
    
    start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        reverseAccess<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
    }
    
    end = high_resolution_clock::now();
    auto reverse_duration = duration_cast<microseconds>(end - start);
    long long avg_reverse = reverse_duration.count() / iterations;
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    bool reverse_correct = verifyResult(h_input, h_output, n, 1);
    
    cout << "\\nAverage execution time: " << avg_reverse << " μs" << endl;
    cout << "Result correctness: " << (reverse_correct ? "✓ Correct" : "✗ Error") << endl;
    
    double bandwidth_reverse = (2.0 * size) / (avg_reverse * 1e-6) / 1e9; // GB/s
    cout << "Memory bandwidth: " << bandwidth_reverse << " GB/s" << endl;
    
    // ===== СРАВНИТЕЛЬНЫЙ АНАЛИЗ =====
    
    cout << "\\n========================================" << endl;
    cout << "PERFORMANCE COMPARISON" << endl;
    cout << "========================================\\n" << endl;
    
    cout << "Access Pattern      | Time (μs) | Bandwidth (GB/s) | vs Coalesced" << endl;
    cout << "--------------------+-----------+------------------+--------------" << endl;
    cout << "Coalesced (stride 1)|  " << avg_coalesced << "     |      " 
         << bandwidth_coalesced << "      |    1.00x ⭐" << endl;
    cout << "Strided (stride 32) |  " << avg_strided << "     |      " 
         << bandwidth_strided << "      |    " 
         << (double)avg_strided / avg_coalesced << "x" << endl;
    cout << "Reverse access      |  " << avg_reverse << "     |      " 
         << bandwidth_reverse << "      |    " 
         << (double)avg_reverse / avg_coalesced << "x" << endl;
    
    double slowdown = (double)avg_strided / avg_coalesced;
    
    cout << "\\n🔥 KEY FINDING:" << endl;
    cout << "Non-coalesced access is " << slowdown << "x SLOWER!" << endl;
    cout << "This demonstrates critical importance of memory access patterns." << endl;
    
    // ===== ОБЪЯСНЕНИЕ И РЕКОМЕНДАЦИИ =====
    
    cout << "\\n========================================" << endl;
    cout << "WHY THIS HAPPENS?" << endl;
    cout << "========================================" << endl;
    
    cout << "\\n📖 Memory Transaction Mechanism:" << endl;
    cout << "1. GPU loads memory in chunks (cache lines, typically 32-128 bytes)" << endl;
    cout << "2. When warp threads access sequential memory:" << endl;
    cout << "   → Single memory transaction loads all needed data" << endl;
    cout << "   → Highly efficient! Maximum bandwidth utilization" << endl;
    cout << "3. When warp threads access scattered memory:" << endl;
    cout << "   → Multiple memory transactions needed" << endl;
    cout << "   → Each transaction may load mostly unused data" << endl;
    cout << "   → Wasted bandwidth and increased latency" << endl;
    
    cout << "\\n💡 Best Practices:" << endl;
    cout << "✓ Array of Structures (AoS) → Structure of Arrays (SoA)" << endl;
    cout << "✓ Use __ldg() for read-only data (L1 cache)" << endl;
    cout << "✓ Align data structures to cache line boundaries" << endl;
    cout << "✓ Profile with NVIDIA Nsight to check memory efficiency" << endl;
    cout << "✓ Aim for >80% memory transaction efficiency" << endl;
    
    cout << "\\n⚠️  Common Pitfalls:" << endl;
    cout << "✗ Random memory access patterns" << endl;
    cout << "✗ Large strides between accesses" << endl;
    cout << "✗ Misaligned data structures" << endl;
    cout << "✗ Accessing scattered elements in warp" << endl;
    
    // Освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    
    cout << "\\n========================================" << endl;
    cout << "Task completed successfully!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
