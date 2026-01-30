#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// Размер матрицы для тестирования
// Matrix dimensions for testing
const int WIDTH = 4096;
const int HEIGHT = 4096;
const int BLOCK_SIZE = 32; // 32x32 threads = 1024 threads per block

// Макрос для проверки ошибок CUDA
// Macro for CUDA error checking
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// 1. Коалесцированный доступ (Эффективный)
// Потоки читают данные последовательно: threadIdx.x меняется быстрее, и соответствует смещению +1 в памяти
// 1. Coalesced Access (Efficient)
// Threads read data sequentially: threadIdx.x varies fastest, corresponding to +1 memory offset
__global__ void coalescedKernel(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x; // Стандартный row-major индекс (Standard row-major index)
        out[idx] = in[idx] * 2.0f; // Простая операция (Simple operation)
    }
}

// 2. Некоалесцированный доступ (Неэффективный)
// Мы меняем индексацию так, что соседние потоки (по threadIdx.x) читают данные с большим шагом (stride)
// 2. Non-coalesced Access (Inefficient)
// We change indexing so adjacent threads (in threadIdx.x) read data with a large stride
__global__ void stridedKernel(const float* in, float* out, int width, int height) {
    // Меняем роль x и y для расчета индекса, но потоки запускаются так же
    // Swap x and y roles for index calculation, but threads launched same way
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Здесь мы представляем, что хотим транспонировать логику доступа или просто читать по столбцам
    // Here we imagine we want to transpose access logic or just read column-wise
    if (x < height && y < width) {
       // Доступ "по столбцам" если представить in как row-major
       // Column-wise access if treating in as row-major
       // Соседние threadIdx.x будут иметь разные 'x', что дает смещение 'width'
       // Adjacent threadIdx.x will have different 'x', resulting in 'width' stride
       int idx = x * width + y; 
       out[idx] = in[idx] * 2.0f; 
    }
}

// 3. Оптимизированная версия с Shared Memory (Транспонирование как пример оптимизации доступа)
// 3. Optimized version with Shared Memory (Matrix Transpose as access optimization example)
__global__ void sharedMemKernel(const float* in, float* out, int width, int height) {
    // В задачах транспонирования часто возникает strided access при записи.
    // Shared memory позволяет читать coalesced, переставлять данные, и писать coalesced (или наоборот).
    // In transpose tasks, strided access often occurs on write.
    // Shared memory allows reading coalesced, rearranging data, and writing coalesced (or vice versa).
    
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 чтобы избежать bank conflicts (avoid bank conflicts)

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Читаем coalesced (строка за строкой)
    // Read coalesced (row by row)
    if (x < width && y < height) {
        int idx_in = y * width + x;
        tile[threadIdx.y][threadIdx.x] = in[idx_in];
    }
    
    __syncthreads();

    // Для демонстрации мы просто пишем обратно, но могли бы изменить паттерн.
    // Если задача просто "обработка", shared memory помогает если есть reuse.
    // Но если задача "показать влияние доступа", shared memory исправляет strided access.
    // Здесь мы просто запишем обратно coalesced, имитируя, что мы данные "упорядочили" в shared mem.
    
    // For demonstration we just write back, but could change pattern.
    // Here we verify computation.
    
    if (x < width && y < height) {
        int idx_out = y * width + x;
        float val = tile[threadIdx.y][threadIdx.x];
        out[idx_out] = val * 2.0f;
    }
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(float);
    int num_elements = WIDTH * HEIGHT;
    
    std::cout << "Matrix Size: " << WIDTH << "x" << HEIGHT << std::endl;
    std::cout << "Elements: " << num_elements << std::endl;

    float *h_in, *h_out;
    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);

    // Инициализация
    // Initialization
    for (int i = 0; i < num_elements; ++i) h_in[i] = 1.0f;

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Настройка сетки
    // Grid Setup
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // События для тайминга
    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float milliseconds = 0;

    // 1. Тест Coalesced
    // 1. Coalesced Test
    cudaEventRecord(start);
    coalescedKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "1. Coalesced Access Time: " << milliseconds << " ms" << std::endl;

    // 2. Тест Strided (Неэффективный)
    // 2. Strided Test (Inefficient)
    // Внимание: Логика доступа внутри ядра изменена, поэтому мы должны понимать, что она делает то же количество работы (чтений/записей)
    // Note: Access logic inside kernel is changed, but work amount (reads/writes) is same
    cudaEventRecord(start);
    stridedKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "2. Strided Access Time:   " << milliseconds << " ms" << std::endl;

    // 3. Тест Shared Memory
    // 3. Shared Memory Test
    cudaEventRecord(start);
    sharedMemKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "3. Shared Memory Time:    " << milliseconds << " ms" << std::endl;

    // Очистка
    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
