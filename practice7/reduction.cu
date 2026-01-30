#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <numeric>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// =========================================================
// CUDA Kernel: Reduction with Shared Memory
// Ядро CUDA: Редукция с использованием разделяемой памяти
// =========================================================
__global__ void reduceSumKernel(float *g_idata, float *g_odata, int n) {
    // Выделяем разделяемую память для блока (размер задается при запуске)
    // Allocate shared memory for the block (size determined at launch)
    extern __shared__ float sdata[];

    // Глобальный идентификатор потока
    // Global thread ID
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Загрузка данных из глобальной памяти в разделяемую
    // Load data from global to shared memory
    // Проверка границ, если n не кратно размеру блока
    // Boundary check if n is not multiple of block size
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    
    // Синхронизация потоков внутри блока, чтобы все данные были загружены
    // Synchronize threads to ensure all data is loaded
    __syncthreads();

    // Выполнение редукции в разделяемой памяти
    // Perform tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // Ждем, пока все потоки закончат текущий шаг
    }

    // Поток 0 записывает результат блока в глобальную память
    // Thread 0 writes the block result to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Функция для обработки ошибок CUDA
// Helper function for checking CUDA errors
void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

// Реализация редукции на CPU
// CPU implementation of reduction
float reduceCPU(const vector<float>& data) {
    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    return sum;
}

// Запуск теста редукции
// Run reduction test
void runTest(int n) {
    cout << "\n------------------------------------------------" << endl;
    cout << "Running Reduction Test with N = " << n << endl;

    // 1. Подготовка данных
    // Prepare data
    vector<float> h_idata(n);
    // Заполняем единицами или случайными числами
    // Fill with 1.0f for easy verification, or random
    for (int i = 0; i < n; i++) h_idata[i] = 1.0f; // rand() / (float)RAND_MAX;

    // Вычисляем эталонное значение на CPU
    // Compute reference value on CPU
    auto start_cpu = high_resolution_clock::now();
    float cpu_sum = reduceCPU(h_idata);
    auto end_cpu = high_resolution_clock::now();
    double time_cpu = duration_cast<microseconds>(end_cpu - start_cpu).count() / 1000.0;

    cout << "CPU Time: " << time_cpu << " ms" << endl;
    cout << "CPU Sum:  " << cpu_sum << endl;

    // 2. Выделение памяти на GPU
    // Allocate GPU memory
    float *d_idata, *d_odata;
    // Определяем размеры сетки
    // Determine grid dimensions
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Выходной массив для частичных сумм блоков
    // Output array for partial sums
    int odata_size = blocks * sizeof(float);

    checkCuda(cudaMalloc(&d_idata, n * sizeof(float)), "Malloc idata");
    checkCuda(cudaMalloc(&d_odata, odata_size), "Malloc odata");

    // 3. Копирование данных на GPU
    // Copy data to GPU
    checkCuda(cudaMemcpy(d_idata, h_idata.data(), n * sizeof(float), cudaMemcpyHostToDevice), "Memcpy To Device");

    // 4. Запуск ядра (Рекурсивная редукция)
    // Launch Kernel (Recursive reduction)
    auto start_gpu = high_resolution_clock::now();

    // Мы можем запускать редукцию в несколько проходов
    // We might need multiple passes
    int current_n = n;
    int current_blocks = blocks;
    
    // Для первого прохода мы берем данные из d_idata
    float *input = d_idata;
    float *output = d_odata;

    while (current_n > 1) {
        // Запуск ядра
        // Launch kernel
        // Shared memory size = threads per block * sizeof(float)
        reduceSumKernel<<<current_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(input, output, current_n);
        checkCuda(cudaGetLastError(), "Kernel Launch");
        
        // Следующий шаг: входной массив теперь - это выход прошлых блоков
        // Next step: input is now the output of previous blocks
        if (current_blocks == 1) break; // Завершили
        
        // Подготовка к следующему проходу
        // Prepare for next pass
        current_n = current_blocks;
        
        // Меняем указатели (в реальном коде можно оптимизировать, чтобы не выделять много памяти)
        // Но здесь проще скопировать результат обратно во входной буфер для следующей итерации или использовать ping-pong
        // Для простоты: используем входной буфер как рабочий, копируем результат в начало input
        // For simplicity: copy partial sums to the beginning of input buffer to reuse it
        checkCuda(cudaMemcpy(d_idata, d_odata, current_blocks * sizeof(float), cudaMemcpyDeviceToDevice), "Memcpy Partial Sums");
        
        input = d_idata;
        current_blocks = (current_n + threads_per_block - 1) / threads_per_block;
        // output (d_odata) переиспользуем
        output = d_odata; 
    }
    
    checkCuda(cudaDeviceSynchronize(), "Device Sync");

    auto end_gpu = high_resolution_clock::now();
    double time_gpu = duration_cast<microseconds>(end_gpu - start_gpu).count() / 1000.0;
    
    // 5. Копирование результата
    // Copy result back
    float gpu_sum = 0.0f;
    // Результат последнего блока лежит в d_odata[0]
    checkCuda(cudaMemcpy(&gpu_sum, d_odata, sizeof(float), cudaMemcpyDeviceToHost), "Memcpy Result");

    cout << "GPU Time: " << time_gpu << " ms" << endl;
    cout << "GPU Sum:  " << gpu_sum << endl;

    // 6. Сравнение
    // Verification
    if (abs(cpu_sum - gpu_sum) < 1e-1) { // Допуск для float
        cout << "Result: CORRECT" << endl;
    } else {
        cout << "Result: INCORRECT (Diff: " << abs(cpu_sum - gpu_sum) << ")" << endl;
    }

    // 7. Очистка
    // Cleanup
    cudaFree(d_idata);
    cudaFree(d_odata);
}

int main() {
    cout << "Practice 7: Parallel Reduction (Sum)" << endl;
    
    runTest(1024);
    runTest(1000000);
    runTest(10000000);

    return 0;
}
