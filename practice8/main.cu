#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// =========================================================
// Задание 1 & 2: Параметры и функции обработки
// =========================================================

// Размер массива (1 000 000 элементов по заданию)
const int N = 1000000;

/**
 * @brief Функция обработки массива на CPU с использованием OpenMP
 * Умножает каждый элемент массива на 2.
 */
void cpu_process(float* data, int size) {
    // Используем OpenMP для распараллеливания цикла на CPU
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        data[i] = data[i] * 2.0f; // Каждое значение умножается на 2
    }
}

/**
 * @brief CUDA Ядро для обработки массива на GPU
 * Умножает каждый элемент массива на 2.
 */
__global__ void gpu_process_kernel(float* data, int size) {
    // Вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверяем, не выходим ли за границы массива
    if (tid < size) {
        data[tid] = data[tid] * 2.0f; // Каждое значение умножается на 2
    }
}

// Вспомогательная функция для проверки ошибок CUDA
void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

// =========================================================
// Главная функция и логика тестирования
// =========================================================

int main() {
    // Выделение памяти на хосте (CPU)
    vector<float> h_data(N);
    vector<float> h_result_cpu(N);
    vector<float> h_result_gpu(N);
    vector<float> h_result_hybrid(N);

    // Инициализация массива случайными данными
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(i);
    }

    cout << fixed << setprecision(3);
    cout << "=========================================================" << endl;
    cout << "Practical Work No. 8: Hybrid CPU/GPU Development" << endl;
    cout << "Array size: " << N << endl;
    cout << "=========================================================" << endl;

    // ---------------------------------------------------------
    // ЗАДАНИЕ 1: Обработка на CPU (OpenMP)
    // ---------------------------------------------------------
    copy(h_data.begin(), h_data.end(), h_result_cpu.begin());
    
    auto start_cpu = high_resolution_clock::now();
    cpu_process(h_result_cpu.data(), N);
    auto end_cpu = high_resolution_clock::now();
    
    double cpu_time = duration_cast<microseconds>(end_cpu - start_cpu).count() / 1000.0;
    cout << "[CPU Only] Time: " << cpu_time << " ms" << endl;

    // ---------------------------------------------------------
    // ЗАДАНИЕ 2: Обработка на GPU (CUDA)
    // ---------------------------------------------------------
    copy(h_data.begin(), h_data.end(), h_result_gpu.begin());
    
    float *d_data;
    checkCuda(cudaMalloc(&d_data, N * sizeof(float)), "Malloc GPU");

    auto start_gpu = high_resolution_clock::now();
    
    // Копирование данных на GPU
    checkCuda(cudaMemcpy(d_data, h_result_gpu.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D");

    // Настройка сетки потоков
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Запуск ядра
    gpu_process_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    checkCuda(cudaDeviceSynchronize(), "Kernel Execution");

    // Копирование обратно на CPU
    checkCuda(cudaMemcpy(h_result_gpu.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
    
    auto end_gpu = high_resolution_clock::now();
    double gpu_time = duration_cast<microseconds>(end_gpu - start_gpu).count() / 1000.0;
    cout << "[GPU Only] Time: " << gpu_time << " ms (including data transfer)" << endl;

    // ---------------------------------------------------------
    // ЗАДАНИЕ 3: Гибридная обработка (CPU + GPU одновременно)
    // ---------------------------------------------------------
    copy(h_data.begin(), h_data.end(), h_result_hybrid.begin());
    
    // Разделяем массив на 2 части (50/50 для простоты)
    int split_idx = N / 2;
    int gpu_part_size = N - split_idx; // Вторая половина на GPU
    int cpu_part_size = split_idx;     // Первая половина на CPU

    auto start_hybrid = high_resolution_clock::now();

    // 1. Асинхронно копируем вторую половину на GPU
    checkCuda(cudaMemcpy(d_data + split_idx, h_result_hybrid.data() + split_idx, gpu_part_size * sizeof(float), cudaMemcpyHostToDevice), "H2D Hybrid");

    // 2. Запускаем GPU ядро (не дожидаясь завершения)
    int hybridBlocks = (gpu_part_size + threadsPerBlock - 1) / threadsPerBlock;
    gpu_process_kernel<<<hybridBlocks, threadsPerBlock>>>(d_data + split_idx, gpu_part_size);

    // 3. В это же время CPU обрабатывает свою часть через OpenMP
    cpu_process(h_result_hybrid.data(), cpu_part_size);

    // 4. Синхронизируем GPU
    checkCuda(cudaDeviceSynchronize(), "Final Sync");

    // 5. Копируем результат с GPU обратно
    checkCuda(cudaMemcpy(h_result_hybrid.data() + split_idx, d_data + split_idx, gpu_part_size * sizeof(float), cudaMemcpyDeviceToHost), "D2H Hybrid");

    auto end_hybrid = high_resolution_clock::now();
    double hybrid_time = duration_cast<microseconds>(end_hybrid - start_hybrid).count() / 1000.0;
    cout << "[Hybrid]   Time: " << hybrid_time << " ms (Both CPU and GPU working)" << endl;

    // ---------------------------------------------------------
    // ЗАДАНИЕ 4: Анализ и проверка корректности
    // ---------------------------------------------------------
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_result_hybrid[i] != h_data[i] * 2.0f) {
            correct = false;
            break;
        }
    }

    cout << "=========================================================" << endl;
    cout << "Status: " << (correct ? "SUCCESS (Results match)" : "FAILURE (Mismatch found)") << endl;
    cout << "Speedup Hybrid vs CPU: " << (cpu_time / hybrid_time) << "x" << endl;
    cout << "Speedup Hybrid vs GPU: " << (gpu_time / hybrid_time) << "x" << endl;
    cout << "=========================================================" << endl;

    // Очистка ресурсов
    cudaFree(d_data);

    return 0;
}
