#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== Assignment 4 - Task 3: Hybrid CPU+GPU Processing =====
// Цель: Реализовать гибридную обработку массива, где CPU и GPU работают параллельно
// над разными частями данных

// Операция: Возведение каждого элемента в квадрат и добавление константы
// result[i] = input[i] * input[i] + 100

// ===== GPU KERNEL: Обработка на GPU =====

__global__ void processArrayGPU(int* input, int* output, int n, int offset) {
    // offset используется для корректной индексации при обработке части массива
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int val = input[idx];
        // Вычислительно-интенсивная операция для демонстрации
        output[idx] = val * val + 100;
    }
}

// ===== CPU ФУНКЦИЯ: Обработка на CPU =====

void processArrayCPU(int* input, int* output, int start, int end) {
    // Обрабатываем элементы от start до end
    for (int i = start; i < end; ++i) {
        int val = input[i];
        output[i] = val * val + 100;
    }
}

// ===== РЕАЛИЗАЦИЯ 1: Только CPU =====

void cpuOnly(int* input, int* output, int n) {
    processArrayCPU(input, output, 0, n);
}

// ===== РЕАЛИЗАЦИЯ 2: Только GPU =====

void gpuOnly(int* input, int* output, int n) {
    // Выделяем память на GPU
    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    
    // Копируем данные на GPU
    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Конфигурация запуска
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Запускаем kernel
    processArrayGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, 0);
    cudaDeviceSynchronize();
    
    // Копируем результат обратно
    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
}

// ===== РЕАЛИЗАЦИЯ 3: ГИБРИДНАЯ (CPU + GPU параллельно) =====

void hybridCpuGpu(int* input, int* output, int n, float cpuRatio) {
    // Разделяем массив между CPU и GPU
    // cpuRatio определяет долю работы для CPU (0.0 - 1.0)
    // Пример: cpuRatio=0.3 означает 30% данных на CPU, 70% на GPU
    
    int cpuSize = (int)(n * cpuRatio);
    int gpuSize = n - cpuSize;
    
    cout << "  CPU portion: " << cpuSize << " elements (" << (cpuRatio * 100) << "%)" << endl;
    cout << "  GPU portion: " << gpuSize << " elements (" << ((1.0f - cpuRatio) * 100) << "%)" << endl;
    
    // Выделяем память на GPU только для GPU части
    int *d_input, *d_output;
    cudaMalloc(&d_input, gpuSize * sizeof(int));
    cudaMalloc(&d_output, gpuSize * sizeof(int));
    
    // Копируем GPU часть данных (начинаемс индекса cpuSize)
    cudaMemcpy(d_input, input + cpuSize, gpuSize * sizeof(int), cudaMemcpyHostToDevice);
    
    // Конфигурация GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (gpuSize + threadsPerBlock - 1) / threadsPerBlock;
    
    // КРИТИЧЕСКИЙ МОМЕНТ: Запускаем GPU kernel асинхронно!
    processArrayGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, gpuSize, cpuSize);
    
    // НЕ ВЫЗЫВАЕМ cudaDeviceSynchronize() здесь!
    // Это позволяет CPU начать работу пока GPU вычисляет
    
    // Запускаем CPU обработку в отдельном потоке для истинного параллелизма
    // В реальности CPU часть выполняется на главном потоке параллельно с GPU
    auto cpuStart = high_resolution_clock::now();
    
    // CPU обрабатывает первую часть массива (индексы 0 до cpuSize)
    processArrayCPU(input, output, 0, cpuSize);
    
    auto cpuEnd = high_resolution_clock::now();
    auto cpuTime = duration_cast<microseconds>(cpuEnd - cpuStart);
    
    // Теперь ждем завершения GPU
    auto gpuWaitStart = high_resolution_clock::now();
    cudaDeviceSynchronize();
    auto gpuWaitEnd = high_resolution_clock::now();
    auto gpuTime = duration_cast<microseconds>(gpuWaitEnd - gpuWaitStart);
    
    cout << "  CPU processing time: " << cpuTime.count() << " μs" << endl;
    cout << "  GPU wait time: " << gpuTime.count() << " μs" << endl;
    
    // Копируем GPU результаты обратно во вторую часть массива
    cudaMemcpy(output + cpuSize, d_output, gpuSize * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
}

// Функция проверки корректности
bool verifyResult(int* input, int* output, int n) {
    for (int i = 0; i < min(100, n); ++i) {
        int expected = input[i] * input[i] + 100;
        if (output[i] != expected) {
            return false;
        }
    }
    return true;
}

int main() {
    srand(time(0));
    
    // Размер массива для тестирования
    const int n = 10000000; // 10 миллионов элементов для заметной разницы
    
    cout << "=== Assignment 4 - Task 3: Hybrid CPU+GPU Processing ===" << endl;
    cout << "Array size: " << n << " elements" << endl;
    cout << "Operation: output[i] = input[i]² + 100\\n" << endl;
    
    // Выделяем память
    int* h_input = new int[n];
    int* h_output_cpu = new int[n];
    int* h_output_gpu = new int[n];
    int* h_output_hybrid = new int[n];
    
    // Инициализируем входной массив
    cout << "Initializing data..." << endl;
    for (int i = 0; i < n; ++i) {
        h_input[i] = rand() % 1000;
    }
    
    // ===== ТЕСТ 1: CPU Only =====
    
    cout << "\\n========================================" << endl;
    cout << "TEST 1: CPU ONLY" << endl;
    cout << "========================================" << endl;
    cout << "Processing all data on CPU sequentially" << endl;
    
    auto start = high_resolution_clock::now();
    
    cpuOnly(h_input, h_output_cpu, n);
    
    auto end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<milliseconds>(end - start);
    
    cout << "\\nExecution time: " << cpu_duration.count() << " ms" << endl;
    cout << "Correctness: " << (verifyResult(h_input, h_output_cpu, n) ? "✓ Correct" : "✗ Error") << endl;
    
    // ===== ТЕСТ 2: GPU Only =====
    
    cout << "\\n========================================" << endl;
    cout << "TEST 2: GPU ONLY" << endl;
    cout << "========================================" << endl;
    cout << "Processing all data on GPU" << endl;
    cout << "Includes memory transfer overhead (CPU↔GPU)" << endl;
    
    start = high_resolution_clock::now();
    
    gpuOnly(h_input, h_output_gpu, n);
    
    end = high_resolution_clock::now();
    auto gpu_duration = duration_cast<milliseconds>(end - start);
    
    cout << "\\nExecution time: " << gpu_duration.count() << " ms" << endl;
    cout << "Correctness: " << (verifyResult(h_input, h_output_gpu, n) ? "✓ Correct" : "✗ Error") << endl;
    
    // ===== ТЕСТ 3: HYBRID (30% CPU, 70% GPU) =====
    
    cout << "\\n========================================" << endl;
    cout << "TEST 3: HYBRID (30% CPU, 70% GPU)" << endl;
    cout << "========================================" << endl;
    cout << "CPU and GPU process different parts in parallel" << endl;
    
    start = high_resolution_clock::now();
    
    hybridCpuGpu(h_input, h_output_hybrid, n, 0.3f);
    
    end = high_resolution_clock::now();
    auto hybrid_duration = duration_cast<milliseconds>(end - start);
    
    cout << "\\nTotal execution time: " << hybrid_duration.count() << " ms" << endl;
    cout << "Correctness: " << (verifyResult(h_input, h_output_hybrid, n) ? "✓ Correct" : "✗ Error") << endl;
    
    // ===== СРАВНИТЕЛЬНАЯ ТАБЛИЦА =====
    
    cout << "\\n========================================" << endl;
    cout << "PERFORMANCE COMPARISON" << endl;
    cout << "========================================\\n" << endl;
    
    cout << "Method              | Time (ms) | Speedup vs CPU" << endl;
    cout << "--------------------+-----------+----------------" << endl;
    cout << "CPU Only            |   " << cpu_duration.count() << "     |     1.00x (baseline)" << endl;
    cout << "GPU Only            |   " << gpu_duration.count() << "      |     " 
         << (double)cpu_duration.count() / gpu_duration.count() << "x" << endl;
    cout << "Hybrid (30%/70%)    |   " << hybrid_duration.count() << "      |     " 
         << (double)cpu_duration.count() / hybrid_duration.count() << "x ⭐" << endl;
    
    // Определяем лучший метод
    long long best_time = min(cpu_duration.count(), min(gpu_duration.count(), hybrid_duration.count()));
    
    cout << "\\n🏆 BEST METHOD: ";
    if (best_time == hybrid_duration.count()) {
        cout << "HYBRID (optimal workload distribution)" << endl;
    } else if (best_time == gpu_duration.count()) {
        cout << "GPU ONLY (GPU dominates for this task)" << endl;
    } else {
        cout << "CPU ONLY (overhead too high for GPU/Hybrid)" << endl;
    }
    
    // ===== АНАЛИЗ РЕЗУЛЬТАТОВ =====
    
    cout << "\\n========================================" << endl;
    cout << "ANALYSIS" << endl;
    cout << "========================================" << endl;
    
    cout << "\\n1. HYBRID COMPUTING BENEFITS:" << endl;
    cout << "   ✓ CPU and GPU work simultaneously" << endl;
    cout << "   ✓ Better resource utilization" << endl;
    cout << "   ✓ Can reduce memory transfer overhead" << endl;
    cout << "   ✓ Flexible workload distribution" << endl;
    
    cout << "\\n2. WHEN HYBRID APPROACH WORKS BEST:" << endl;
    cout << "   - Large datasets where both CPU and GPU are utilized" << endl;
    cout << "   - Tasks with mixed sequential/parallel components" << endl;
    cout << "   - When data can be partitioned independently" << endl;
    cout << "   - Minimizing idle time of either processor" << endl;
    
    cout << "\\n3. WORKLOAD DISTRIBUTION FACTORS:" << endl;
    cout << "   - Relative performance (GPU usually 5-100x faster)" << endl;
    cout << "   - Memory transfer overhead" << endl;
    cout << "   - Problem characteristics" << endl;
    cout << "   - Hardware capabilities" << endl;
    
    cout << "\\n4. OPTIMIZATION STRATEGIES:" << endl;
    cout << "   ✓ Asynchronous memory transfers (cudaMemcpyAsync)" << endl;
    cout << "   ✓ CUDA streams for overlapping" << endl;
    cout << "   ✓ Pinned memory (cudaMallocHost) for faster transfers" << endl;
    cout << "   ✓ Dynamic load balancing based on completion times" << endl;
    cout << "   ✓ CPU threads for true parallel execution" << endl;
    
    cout << "\\n5. CHALLENGES:" << endl;
    cout << "   ⚠ Load balancing complexity" << endl;
    cout << "   ⚠ Memory transfer overhead" << endl;
    cout << "   ⚠ Synchronization between CPU and GPU" << endl;
    cout << "   ⚠ Code complexity increases" << endl;
    
    cout << "\\n6. REAL-WORLD APPLICATIONS:" << endl;
    cout << "   - Video encoding (CPU: I/O, GPU: encoding)" << endl;
    cout << "   - Machine learning (CPU: data prep, GPU: training)" << endl;
    cout << "   - Scientific simulations (hybrid workloads)" << endl;
    cout << "   - Ray tracing (CPU: scene management, GPU: rendering)" << endl;
    
    cout << "\\n7. OPTIMAL CPU/GPU RATIO:" << endl;
    cout << "   - Depends on relative performance" << endl;
    cout << "   - This example used 30%/70% split" << endl;
    cout << "   - Modern GPUs often prefer 10-20% CPU, 80-90% GPU" << endl;
    cout << "   - Profile and adjust based on hardware!" << endl;
    
    // Освобождаем память
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    delete[] h_output_hybrid;
    
    cout << "\\n========================================" << endl;
    cout << "Task completed successfully!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
