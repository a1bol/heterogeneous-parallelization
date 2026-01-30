#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int BLOCK_SIZE = 256;

// =========================================================
// CUDA Kernels: Scan (Prefix Sum)
// Ядра CUDA: Сканирование (Префиксная сумма)
// =========================================================

// Ядро 1: Сканирование внутри блока (Hillis-Steele для простоты или Blelloch)
// Kernel 1: Block-level scan
// Сохраняет сумму блока в aux_array (для многоблочного сканирования)
__global__ void prescanBlockKernel(float *g_odata, float *g_idata, float *aux_array, int n) {
    // В разделяемой памяти храним данные блока
    // Shared memory for block data
    __shared__ float temp[BLOCK_SIZE]; 

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Загрузка данных (включая проверку границ)
    // Load data (with bound check)
    if (gid < n) {
        temp[tid] = g_idata[gid];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Алгоритм Хиллеса-Стила (Hillis-Steele Scan)
    // Простой для реализации, хотя делает больше работы (O(N log N)) чем Blelloch (O(N))
    // Simple implementation
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        float val = 0.0f;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    // Запись результата в глобальную память
    // Write result to global memory
    if (gid < n) {
        // Для инклюзивного скана (inclusive scan): просто пишем temp[tid]
        // Для эксклюзивного (exclusive scan): temp[tid] - g_idata[gid] или сдвиг
        // Для задачи обычно нужен inclusive prefix sum [1, 3, 6...]
        g_odata[gid] = temp[tid];
    }

    // Последний поток сохраняет общую сумму блока в вспомогательный массив
    // Last thread saves block sum to aux array
    // Это нужно для связи блоков
    if (tid == BLOCK_SIZE - 1 && aux_array != NULL) {
        aux_array[blockIdx.x] = temp[tid];
    }
}

// Ядро 2: Добавление базового значения к блоку (Uniform Add)
// Kernel 2: Add base value to block
__global__ void addBlockSumKernel(float *g_data, float *aux_array, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Блок 0 не нуждается в добавке
    if (blockIdx.x > 0 && gid < n) {
        // aux_array хранит префиксные суммы блоков
        // aux_array stores scanned block sums
        // Нам нужно добавить значение из ПРЕДЫДУЩЕГО элемента aux_array (эксклюзивно)
        // То есть для блока K мы добавляем aux_array[K-1]
        
        g_data[gid] += aux_array[blockIdx.x - 1];
    }
}

// Helper: Check CUDA error
void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

// CPU Scan
void scanCPU(const vector<float>& input, vector<float>& output) {
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); i++) {
        sum += input[i];
        output[i] = sum;
    }
}

// Функция для запуска многоблочного сканирования
// Function to run multi-block scan
void recursiveScan(float *d_out, float *d_in, int n) {
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    // 1. Если массив маленький (1 блок), просто сканируем
    if (blocks == 1) {
        prescanBlockKernel<<<blocks, threads_per_block>>>(d_out, d_in, NULL, n);
        return;
    }

    // 2. Если массив большой:
    // a. Выделяем память под суммы блоков
    float *d_aux;   // Входные суммы блоков
    float *d_aux_scanned; // Просканированные суммы блоков
    checkCuda(cudaMalloc(&d_aux, blocks * sizeof(float)), "Malloc Aux");
    checkCuda(cudaMalloc(&d_aux_scanned, blocks * sizeof(float)), "Malloc Aux Scanned");

    // b. Сканируем каждый блок и сохраняем суммы в d_aux
    prescanBlockKernel<<<blocks, threads_per_block>>>(d_out, d_in, d_aux, n);
    checkCuda(cudaGetLastError(), "Scan Phase 1");

    // c. Рекурсивно сканируем массив сумм (d_aux -> d_aux_scanned)
    recursiveScan(d_aux_scanned, d_aux, blocks);

    // d. Добавляем просканированные суммы к элементам блоков
    addBlockSumKernel<<<blocks, threads_per_block>>>(d_out, d_aux_scanned, n);
    checkCuda(cudaGetLastError(), "Scan Phase 3");

    // Очистка временной памяти
    cudaFree(d_aux);
    cudaFree(d_aux_scanned);
}

void runTest(int n) {
    cout << "\n------------------------------------------------" << endl;
    cout << "Running Scan Test with N = " << n << endl;

    // Data
    vector<float> h_in(n, 1.0f); // Заполняем единицами
    vector<float> h_out_cpu(n);
    vector<float> h_out_gpu(n);

    // CPU Pass
    auto start_cpu = high_resolution_clock::now();
    scanCPU(h_in, h_out_cpu);
    auto end_cpu = high_resolution_clock::now();
    double time_cpu = duration_cast<microseconds>(end_cpu - start_cpu).count() / 1000.0;
    cout << "CPU Time: " << time_cpu << " ms" << endl;

    // GPU Pass
    float *d_in, *d_out;
    checkCuda(cudaMalloc(&d_in, n * sizeof(float)), "Malloc In");
    checkCuda(cudaMalloc(&d_out, n * sizeof(float)), "Malloc Out");

    checkCuda(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice), "Memcpy In");

    auto start_gpu = high_resolution_clock::now();
    
    // Запускаем рекурсивный скан
    recursiveScan(d_out, d_in, n);
    checkCuda(cudaDeviceSynchronize(), "Device Sync");

    auto end_gpu = high_resolution_clock::now();
    double time_gpu = duration_cast<microseconds>(end_gpu - start_gpu).count() / 1000.0;
    cout << "GPU Time: " << time_gpu << " ms" << endl;

    // Verify
    checkCuda(cudaMemcpy(h_out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy Out");

    // Проверка (первый, последний и случайные элементы)
    bool correct = true;
    if (abs(h_out_gpu[0] - h_out_cpu[0]) > 0.1f) correct = false;
    if (abs(h_out_gpu[n-1] - h_out_cpu[n-1]) > 0.1f) correct = false;
    
    // Доп проверка
    for (int i = 0; i < min(n, 100); i++) {
        if (abs(h_out_gpu[i] - h_out_cpu[i]) > 0.1f) {
            correct = false; 
            cout << "Mismatch at " << i << ": GPU " << h_out_gpu[i] << " != CPU " << h_out_cpu[i] << endl;
            break;
        }
    }

    if (correct) {
        cout << "Result: CORRECT" << endl;
        cout << "Sample [Last]: " << h_out_gpu[n-1] << " (Expected: " << n << ".0)" << endl;
    } else {
        cout << "Result: INCORRECT" << endl;
        cout << "Sample [Last]: " << h_out_gpu[n-1] << " (Expected: " << h_out_cpu[n-1] << ")" << endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    cout << "Practice 7: Parallel Scan (Prefix Sum)" << endl;
    
    runTest(1024);
    runTest(1000000);
    // Для 10M может потребоваться увеличение стека или кучи GPU, или просто будет чуть дольше
    // For 10M it works fine with global memory recursion
    runTest(10000000);

    return 0;
}
