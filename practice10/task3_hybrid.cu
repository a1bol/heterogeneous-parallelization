#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>

// Размер данных: 50 миллионов
// Data size: 50 million
const int N = 50000000;
const int BLOCK_SIZE = 256;

// Макрос проверки ошибок
// Error check macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// Функция "тяжелых" вычислений для CPU
// Heavy computation function for CPU
void cpu_process(float* data, int start, int end) {
    #pragma omp parallel for
    for (int i = start; i < end; ++i) {
        data[i] = sqrt(sin(data[i]) * cos(data[i]) + 1.0f);
    }
}

// Ядро для GPU
// GPU Kernel
__global__ void gpu_kernel(float* data, int offset, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx]; // data указывает на device буфер (data points to device buffer)
        // Те же вычисления что на CPU
        // Same calculation as CPU
        data[idx] = sqrtf(sinf(val) * cosf(val) + 1.0f);
    }
}

int main() {
    float *h_data;
    // pinned memory для асинхронных операций
    // pinned memory for async operations
    CHECK_CUDA(cudaMallocHost((void**)&h_data, N * sizeof(float)));

    // Инициализация
    // Initialization
    for (int i = 0; i < N; ++i) h_data[i] = (float)i;

    std::cout << "Data size: " << N << " elements" << std::endl;

    // Разделение работы
    // Workload split
    float gpu_ratio = 0.5f; // 50% GPU, 50% CPU
    int gpu_count = N * gpu_ratio;
    int cpu_count = N - gpu_count;
    
    std::cout << "GPU Load: " << gpu_count << ", CPU Load: " << cpu_count << std::endl;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, gpu_count * sizeof(float)));

    // Создание потока
    // Create stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Тайминг
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 1. Запуск GPU части (Асинхронно)
    // 1. Launch GPU part (Async)
    // Копирование H2D
    // Copy H2D
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, gpu_count * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Запуск ядра
    // Launch kernel
    int grid_size = (gpu_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_data, 0, gpu_count);

    // Копирование D2H
    // Copy D2H
    CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, gpu_count * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // 2. Выполнение CPU части (Пока GPU работает)
    // 2. Execute CPU part (While GPU is working)
    // CPU берет вторую половину массива
    // CPU takes the second half of the array
    cpu_process(h_data, gpu_count, N);

    // 3. Синхронизация
    // 3. Synchronization
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Total Hybrid Execution Time: " << milliseconds << " ms" << std::endl;

    // Очистка
    // Cleanup
    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
