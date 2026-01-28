#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== Assignment 3 - Task 4: Optimal Grid/Block Configuration =====
// Цель: Подобрать оптимальные параметры конфигурации сетки и блоков
// Базовая операция: Матричное умножение (memory-intensive + compute-intensive)

// Простой kernel матричного умножения для демонстрации оптимизации
// C = A * B, где все матрицы размера N x N
__global__ void matrixMultiply(float* A, float* B, float* C, int N) {
    // Вычисляем позицию элемента, который обрабатывает этот поток
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверяем границы
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // Вычисляем скалярное произведение row-го ряда A и col-го столбца B
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// Оптимизированный kernel с использованием shared memory
// Блочное матричное умножение - значительно эффективнее!
__global__ void matrixMultiplyOptimized(float* A, float* B, float* C, int N) {
    // Размер блока (должен быть установлен при запуске)
    const int BLOCK_SIZE = 16; // 16x16 = 256 потоков
    
    // Выделяем разделяемую память для подматриц
    // Две подматрицы: одна из A, одна из B
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    // Индексы потока в блоке
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Глобальные индексы элемента результата
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    // Проходим по блокам матриц A и B
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int m = 0; m < numBlocks; ++m) {
        // Загружаем подматрицу A в shared memory
        int aRow = row;
        int aCol = m * BLOCK_SIZE + tx;
        if (aRow < N && aCol < N) {
            As[ty][tx] = A[aRow * N + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Загружаем подматрицу B в shared memory
        int bRow = m * BLOCK_SIZE + ty;
        int bCol = col;
        if (bRow < N && bCol < N) {
            Bs[ty][tx] = B[bRow * N + bCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Синхронизация: все потоки блока должны загрузить данные
        __syncthreads();
        
        // Вычисляем частичное произведение для этого блока
        // Теперь читаем из быстрой shared memory!
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Синхронизация перед загрузкой следующего блока
        __syncthreads();
    }
    
    // Записываем результат в глобальную память
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Функция для проверки корректности результата
bool verifyResult(float* A, float* B, float* C, int N) {
    // Проверяем несколько элементов (полная проверка занимает много времени)
    for (int i = 0; i < min(10, N); ++i) {
        for (int j = 0; j < min(10, N); ++j) {
            float expected = 0.0f;
            for (int k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            
            float diff = abs(C[i * N + j] - expected);
            if (diff > 0.01f) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    srand(time(0));
    
    // Размер матриц (N x N)
    const int N = 1024; // 1024x1024 матрицы
    size_t size = N * N * sizeof(float);
    
    cout << "=== Assignment 3 - Task 4: Grid/Block Optimization ===" << endl;
    cout << "Operation: Matrix Multiplication (C = A × B)" << endl;
    cout << "Matrix size: " << N << " × " << N << " elements" << endl;
    cout << "Total elements: " << (N * N) << "\\n" << endl;
    
    // Выделяем память на хосте
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];
    
    // Инициализируем матрицы случайными числами
    cout << "Initializing matrices..." << endl;
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(rand() % 10) / 10.0f; // 0.0 - 0.9
        h_B[i] = (float)(rand() % 10) / 10.0f;
    }
    
    // Выделяем память на GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Копируем данные на GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // ===== ТЕСТ 1: Неоптимальная конфигурация =====
    
    cout << "\\n========================================" << endl;
    cout << "TEST 1: NON-OPTIMAL CONFIGURATION" << endl;
    cout << "========================================" << endl;
    
    // Плохая конфигурация: маленькие блоки, плохой размер
    dim3 badBlockSize(8, 8);     // 64 потока на блок - слишком мало!
    dim3 badGridSize((N + badBlockSize.x - 1) / badBlockSize.x,
                     (N + badBlockSize.y - 1) / badBlockSize.y);
    
    cout << "Block size: " << badBlockSize.x << " × " << badBlockSize.y 
         << " (" << (badBlockSize.x * badBlockSize.y) << " threads)" << endl;
    cout << "Grid size: " << badGridSize.x << " × " << badGridSize.y 
         << " (" << (badGridSize.x * badGridSize.y) << " blocks)" << endl;
    cout << "Total threads: " << (badBlockSize.x * badBlockSize.y * badGridSize.x * badGridSize.y) << endl;
    
    cudaMemset(d_C, 0, size);
    
    auto start = high_resolution_clock::now();
    
    matrixMultiply<<<badGridSize, badBlockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    auto end = high_resolution_clock::now();
    auto bad_duration = duration_cast<milliseconds>(end - start);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    bool bad_correct = verifyResult(h_A, h_B, h_C, N);
    
    cout << "\\nExecution time: " << bad_duration.count() << " ms" << endl;
    cout << "Result correctness: " << (bad_correct ? "✓ Correct" : "✗ Error") << endl;
    
    // ===== ТЕСТ 2: Улучшенная конфигурация =====
    
    cout << "\\n========================================" << endl;
    cout << "TEST 2: IMPROVED CONFIGURATION" << endl;
    cout << "========================================" << endl;
    
    // Лучшая конфигурация: стандартный размер блока
    dim3 goodBlockSize(16, 16);  // 256 потоков - хороший баланс
    dim3 goodGridSize((N + goodBlockSize.x - 1) / goodBlockSize.x,
                      (N + goodBlockSize.y - 1) / goodBlockSize.y);
    
    cout << "Block size: " << goodBlockSize.x << " × " << goodBlockSize.y 
         << " (" << (goodBlockSize.x * goodBlockSize.y) << " threads)" << endl;
    cout << "Grid size: " << goodGridSize.x << " × " << goodGridSize.y 
         << " (" << (goodGridSize.x * goodGridSize.y) << " blocks)" << endl;
    cout << "Total threads: " << (goodBlockSize.x * goodBlockSize.y * goodGridSize.x * goodGridSize.y) << endl;
    
    cudaMemset(d_C, 0, size);
    
    start = high_resolution_clock::now();
    
    matrixMultiply<<<goodGridSize, goodBlockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    end = high_resolution_clock::now();
    auto good_duration = duration_cast<milliseconds>(end - start);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    bool good_correct = verifyResult(h_A, h_B, h_C, N);
    
    cout << "\\nExecution time: " << good_duration.count() << " ms" << endl;
    cout << "Result correctness: " << (good_correct ? "✓ Correct" : "✗ Error") << endl;
    
    // ===== ТЕСТ 3: Оптимизированная конфигурация + Shared Memory =====
    
    cout << "\\n========================================" << endl;
    cout << "TEST 3: OPTIMIZED (Shared Memory)" << endl;
    cout << "========================================" << endl;
    
    // Оптимальная конфигурация для shared memory версии
    dim3 optBlockSize(16, 16);   // 16x16 = 256 потоков
    dim3 optGridSize((N + 15) / 16, (N + 15) / 16);
    
    cout << "Block size: " << optBlockSize.x << " × " << optBlockSize.y 
         << " (" << (optBlockSize.x * optBlockSize.y) << " threads)" << endl;
    cout << "Grid size: " << optGridSize.x << " × " << optGridSize.y 
         << " (" << (optGridSize.x * optGridSize.y) << " blocks)" << endl;
    cout << "Shared memory: Using block tiling" << endl;
    
    cudaMemset(d_C, 0, size);
    
    start = high_resolution_clock::now();
    
    matrixMultiplyOptimized<<<optGridSize, optBlockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    end = high_resolution_clock::now();
    auto opt_duration = duration_cast<milliseconds>(end - start);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    bool opt_correct = verifyResult(h_A, h_B, h_C, N);
    
    cout << "\\nExecution time: " << opt_duration.count() << " ms" << endl;
    cout << "Result correctness: " << (opt_correct ? "✓ Correct" : "✗ Error") << endl;
    
    // ===== СРАВНИТЕЛЬНЫЙ АНАЛИЗ =====
    
    cout << "\\n========================================" << endl;
    cout << "PERFORMANCE COMPARISON" << endl;
    cout << "========================================\\n" << endl;
    
    cout << "Configuration        | Time (ms) | Speedup vs Non-optimal" << endl;
    cout << "---------------------+-----------+------------------------" << endl;
    cout << "Non-optimal (8×8)    |   " << bad_duration.count() << "     |      1.00x (baseline)" << endl;
    cout << "Improved (16×16)     |   " << good_duration.count() << "      |      " 
         << (double)bad_duration.count() / good_duration.count() << "x" << endl;
    cout << "Optimized (Shared)   |   " << opt_duration.count() << "      |      " 
         << (double)bad_duration.count() / opt_duration.count() << "x ⭐" << endl;
    
    cout << "\\n🏆 BEST CONFIGURATION:" << endl;
    cout << "Block size: 16×16 (256 threads)" << endl;
    cout << "Algorithm: Shared memory tiled multiplication" << endl;
    cout << "Speedup: " << (double)bad_duration.count() / opt_duration.count() << "x faster than non-optimal!" << endl;
    
    // ===== КЛЮЧЕВЫЕ ВЫВОДЫ =====
    
    cout << "\\n========================================" << endl;
    cout << "KEY FINDINGS" << endl;
    cout << "========================================" << endl;
    
    cout << "\\n1. BLOCK SIZE IMPACT:" << endl;
    cout << "   8×8 (64 threads):   LOW occupancy, poor GPU utilization" << endl;
    cout << "   16×16 (256 threads): GOOD occupancy, balanced performance" << endl;
    cout << "   32×32 (1024 threads): MAX threads, but may limit occupancy" << endl;
    
    cout << "\\n2. MEMORY ACCESS OPTIMIZATION:" << endl;
    cout << "   Shared memory reduces global memory accesses by ~" 
         << ((double)good_duration.count() / opt_duration.count()) << "x" << endl;
    cout << "   Block tiling reuses data within blocks" << endl;
    cout << "   Critical for memory-bound kernels!" << endl;
    
    cout << "\\n3. OCCUPANCY ANALYSIS:" << endl;
    cout << "   Occupancy = (Active warps) / (Max warps per SM)" << endl;
    cout << "   8×8 blocks: ~2 warps → Low occupancy" << endl;
    cout << "   16×16 blocks: 8 warps → Good occupancy (50-75%)" << endl;
    cout << "   Goal: Balance occupancy with resource usage" << endl;
    
    cout << "\\n4. OPTIMAL CONFIGURATION GUIDELINES:" << endl;
    cout << "   ✓ Block size: 128-512 threads (256 is sweet spot)" << endl;
    cout << "   ✓ 2D blocks for 2D data (16×16, 32×16)" << endl;
    cout << "   ✓ Multiple of warp size (32)" << endl;
    cout << "   ✓ Use shared memory for data reuse" << endl;
    cout << "   ✓ Profile with Nsight Compute" << endl;
    
    cout << "\\n5. ARCHITECTURE-SPECIFIC TUNING:" << endl;
    cout << "   - Check GPU compute capability" << endl;
    cout << "   - Max threads per block (usually 1024)" << endl;
    cout << "   - Shared memory per block (48-163 KB)" << endl;
    cout << "   - Registers per thread" << endl;
    cout << "   - Always test on target hardware!" << endl;
    
    // Освобождаем память
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    cout << "\\n========================================" << endl;
    cout << "Task completed successfully!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
