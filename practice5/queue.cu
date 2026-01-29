#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// =========================================================
// Структура Очередь для работы на GPU
// =========================================================
struct Queue {
    int *data;      // Указатель на массив данных
    int head;       // Индекс головы (откуда читаем)
    int tail;       // Индекс хвоста (куда пишем)
    int capacity;   // Максимальная ёмкость

    // Инициализация очереди (вызывается на устройстве)
    __device__ void init(int *buffer, int size) {
        data = buffer;
        head = 0;       // Начало очереди
        tail = 0;       // Конец очереди
        capacity = size;
    }

    // Операция Enqueue (добавление)
    __device__ bool enqueue(int value) {
        // Атомарно увеличиваем tail, резервируя позицию для записи
        // atomicAdd возвращает СТАРОЕ значение tail
        int pos = atomicAdd(&tail, 1);

        // Проверяем, не вышли ли за пределы массива
        if (pos < capacity) {
            data[pos] = value; // Записываем значение в зарезервированную ячейку
            return true;
        } 
        // Если очередь полна, ничего не делаем (позиция пропадает, но память не портится)
        return false;
    }

    // Операция Dequeue (извлечение)
    __device__ bool dequeue(int *value) {
        // Атомарно увеличиваем head, резервируя позицию для чтения
        // atomicAdd возвращает СТАРОЕ значение head
        int pos = atomicAdd(&head, 1);

        // Проверяем валидность позиции:
        // 1. pos должен быть меньше capacity (чтобы не читать за границей памяти)
        // 2. pos должен быть меньше tail (чтобы не читать то, что еще не записано)
        // Внимание: чтение tail здесь не атомарно, но так как tail только растет,
        // риск прочитать "старое" значение безопасен (просто скажем, что пусто),
        // но есть риск гонки, если enqueue еще пишет данные.
        // Для учебного примера считаем, что данные уже записаны (барьер между фазами)
        
        if (pos < capacity && pos < tail) {
            *value = data[pos]; // Читаем значение
            return true;
        }
        
        // Если очередь пуста или вышли за границы
        return false;
    }
    
    // Получение количества элементов (примерно)
    __device__ int size() {
        return tail - head;
    }
};

// =========================================================
// Ядра CUDA (Kernels)
// =========================================================

// Ядро для инициализации очереди
__global__ void initQueueKernel(Queue *q, int *buffer, int capacity) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        q->init(buffer, capacity);
    }
}

// Ядро для параллельного добавления (Enqueue)
__global__ void enqueueTaskKernel(Queue *q, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_elements) {
        // Каждый поток добавляет свой ID
        q->enqueue(tid);
    }
}

// Ядро для параллельного извлечения (Dequeue)
__global__ void dequeueTaskKernel(Queue *q, int num_elements, long long *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_elements) {
        int val;
        // Пытаемся извлечь элемент
        if (q->dequeue(&val)) {
            // Если успешно, добавляем к контрольной сумме
            atomicAdd((unsigned long long*)sum, (unsigned long long)val);
        }
    }
}

// =========================================================
// Вспомогательные функции
// =========================================================

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// =========================================================
// Main
// =========================================================

int main() {
    // Параметры теста
    const int NUM_ELEMENTS = 1000000; // 1 миллион элементов
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cout << "=== Practice 5: Parallel Queue on CUDA ===" << endl;
    cout << "Elements to process: " << NUM_ELEMENTS << endl;
    
    // 1. Выделение памяти
    Queue *d_queue;       // Указатель на структуру очереди
    int *d_buffer;        // Буфер данных
    long long *d_sum;     // Переменная для суммы

    checkCudaError(cudaMalloc(&d_queue, sizeof(Queue)), "Malloc Queue");
    checkCudaError(cudaMalloc(&d_buffer, NUM_ELEMENTS * sizeof(int)), "Malloc Buffer");
    checkCudaError(cudaMalloc(&d_sum, sizeof(long long)), "Malloc Sum");
    
    checkCudaError(cudaMemset(d_sum, 0, sizeof(long long)), "Memset Sum");

    // 2. Инициализация
    initQueueKernel<<<1, 1>>>(d_queue, d_buffer, NUM_ELEMENTS);
    checkCudaError(cudaDeviceSynchronize(), "Init Queue");

    // 3. Тест ENQUEUE
    cout << "\n[Enqueue Test] Starting..." << endl;
    auto start = high_resolution_clock::now();

    enqueueTaskKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_queue, NUM_ELEMENTS);
    checkCudaError(cudaDeviceSynchronize(), "Enqueue Kernel");

    auto end = high_resolution_clock::now();
    double enq_time = duration_cast<milliseconds>(end - start).count();
    cout << "[Enqueue Test] Completed in " << enq_time << " ms" << endl;
    cout << "Throughput: " << (NUM_ELEMENTS / (enq_time / 1000.0)) / 1e6 << " Mops/sec" << endl;

    // 4. Тест DEQUEUE
    cout << "\n[Dequeue Test] Starting..." << endl;
    start = high_resolution_clock::now();

    dequeueTaskKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_queue, NUM_ELEMENTS, d_sum);
    checkCudaError(cudaDeviceSynchronize(), "Dequeue Kernel");

    end = high_resolution_clock::now();
    double deq_time = duration_cast<milliseconds>(end - start).count();
    cout << "[Dequeue Test] Completed in " << deq_time << " ms" << endl;
    cout << "Throughput: " << (NUM_ELEMENTS / (deq_time / 1000.0)) / 1e6 << " Mops/sec" << endl;

    // 5. Проверка результатов
    long long h_sum = 0;
    cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    
    long long expected_sum = (long long)NUM_ELEMENTS * (NUM_ELEMENTS - 1) / 2;
    
    cout << "\n[Verification]" << endl;
    cout << "Sum of dequeued elements: " << h_sum << endl;
    cout << "Expected sum:             " << expected_sum << endl;
    
    if (h_sum == expected_sum) {
        cout << "Status: SUCCESS (Queue works correctly)" << endl;
    } else {
        cout << "Status: FAILURE (Sum mismatch)" << endl;
    }

    // 6. Очистка
    cudaFree(d_queue);
    cudaFree(d_buffer);
    cudaFree(d_sum);

    return 0;
}
