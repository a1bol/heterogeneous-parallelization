#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// =========================================================
// Структура Стек для работы на GPU
// =========================================================
struct Stack {
    int *data;      // Указатель на массив данных
    int top;        // Индекс вершины стека
    int capacity;   // Максимальная ёмкость

    // Инициализация стека (вызывается на устройстве)
    __device__ void init(int *buffer, int size) {
        data = buffer;
        top = -1;       // Стек пуст, top указывает "перед" первым элементом
        capacity = size;
    }

    // Операция Push (добавление) с использованием атомиков
    __device__ bool push(int value) {
        // Атомарно увеличиваем top и получаем старое значение
        // top + 1 - это позиция для записи нового элемента
        // atomicAdd возвращает СТАРОЕ значение, поэтому нам нужно +1 для индекса записи?
        // Нет, обычно atomicAdd(&top, 1) возвращает старый top.
        // Если top был -1, atomicAdd вернет -1, новый top станет 0.
        // Значит позиция для записи = old_top + 1.
        
        int old_top = atomicAdd(&top, 1);
        int pos = old_top + 1;

        if (pos < capacity) {
            data[pos] = value; // Записываем значение в безопасную позицию
            return true;
        } else {
            // Если вышли за границы, возвращаем счетчик назад (опционально, для корректности состояния)
            // Но в простой реализации при переполнении просто отказываем
            atomicSub(&top, 1); 
            return false;
        }
    }

    // Операция Pop (удаление) с использованием атомиков
    __device__ bool pop(int *value) {
        // Атомарно уменьшаем top
        // atomicSub возвращает СТАРОЕ значение.
        // Мы пытаемся забрать элемент с текущего top.
        // Но есть нюанс гонки: если два потока читают top, один может забрать, а другой прочитать мусор.
        // Безопасная схема: сначала резервируем индекс (уменьшаем), потом читаем.
        
        int old_top = atomicSub(&top, 1);
        
        // old_top - это индекс элемента, который мы "забираем"
        if (old_top > -1) {
            *value = data[old_top];
            return true;
        } else {
            // Стек пуст, восстанавливаем значение (чтобы top не уходил далеко в минус)
            atomicAdd(&top, 1);
            return false;
        }
    }
    
    // Получение текущего размера (для отладки)
    __device__ int size() {
        return top + 1;
    }
};

// =========================================================
// Ядра CUDA (Kernels)
// =========================================================

// Ядро для инициализации стека
__global__ void initStackKernel(Stack *stack, int *buffer, int capacity) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        stack->init(buffer, capacity);
    }
}

// Ядро для параллельного добавления (Push)
// Каждый поток пытается добавить свой глобальный ID
__global__ void pushTaskKernel(Stack *stack, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_elements) {
        // Пытаемся добавить tid в стек
        bool success = stack->push(tid);
        
        // В реальной задаче можно обработать неудачу (например, стек полон)
        // Но здесь мы просто тестируем скорость
    }
}

// Ядро для параллельного извлечения (Pop)
__global__ void popTaskKernel(Stack *stack, int num_elements, long long *sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_elements) {
        int val;
        if (stack->pop(&val)) {
            // Если успешно извлекли, можно что-то сделать с данными
            // Например, атомарно добавить к сумме (только для проверки)
            atomicAdd((unsigned long long*)sum, (unsigned long long)val);
        }
    }
}

// =========================================================
// Вспомогательные функции
// =========================================================

// Проверка ошибок CUDA
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

    cout << "=== Practice 5: Parallel Stack on CUDA ===" << endl;
    cout << "Elements to process: " << NUM_ELEMENTS << endl;
    cout << "Threads per block: " << BLOCK_SIZE << endl;
    cout << "Blocks: " << GRID_SIZE << endl;

    // 1. Выделение памяти
    Stack *d_stack;       // Указатель на структуру стека на GPU
    int *d_buffer;        // Буфер данных для стека на GPU
    long long *d_sum;     // Переменная для суммы (проверка целостности)

    checkCudaError(cudaMalloc(&d_stack, sizeof(Stack)), "Malloc Stack");
    checkCudaError(cudaMalloc(&d_buffer, NUM_ELEMENTS * sizeof(int)), "Malloc Buffer");
    checkCudaError(cudaMalloc(&d_sum, sizeof(long long)), "Malloc Sum");
    
    // Обнуляем сумму
    checkCudaError(cudaMemset(d_sum, 0, sizeof(long long)), "Memset Sum");

    // 2. Инициализация стека
    initStackKernel<<<1, 1>>>(d_stack, d_buffer, NUM_ELEMENTS);
    checkCudaError(cudaDeviceSynchronize(), "Init Stack");

    // 3. Тест PUSH
    cout << "\n[Push Test] Starting..." << endl;
    auto start = high_resolution_clock::now();

    pushTaskKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_stack, NUM_ELEMENTS);
    checkCudaError(cudaDeviceSynchronize(), "Push Kernel");

    auto end = high_resolution_clock::now();
    double push_time = duration_cast<milliseconds>(end - start).count();
    cout << "[Push Test] Completed in " << push_time << " ms" << endl;
    cout << "Throughput: " << (NUM_ELEMENTS / (push_time / 1000.0)) / 1e6 << " Mops/sec" << endl;

    // 4. Тест POP
    cout << "\n[Pop Test] Starting..." << endl;
    start = high_resolution_clock::now();

    popTaskKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_stack, NUM_ELEMENTS, d_sum);
    checkCudaError(cudaDeviceSynchronize(), "Pop Kernel");

    end = high_resolution_clock::now();
    double pop_time = duration_cast<milliseconds>(end - start).count();
    cout << "[Pop Test] Completed in " << pop_time << " ms" << endl;
    cout << "Throughput: " << (NUM_ELEMENTS / (pop_time / 1000.0)) / 1e6 << " Mops/sec" << endl;

    // 5. Проверка результатов
    long long h_sum = 0;
    cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    
    // Ожидаемая сумма чисел от 0 до N-1: N*(N-1)/2
    long long expected_sum = (long long)NUM_ELEMENTS * (NUM_ELEMENTS - 1) / 2;
    
    cout << "\n[Verification]" << endl;
    cout << "Sum of popped elements: " << h_sum << endl;
    cout << "Expected sum:           " << expected_sum << endl;
    
    if (h_sum == expected_sum) {
        cout << "Status: SUCCESS (All elements pushed and popped correctly)" << endl;
    } else {
        cout << "Status: FAILURE (Sum mismatch)" << endl;
    }

    // 6. Очистка ресурсов
    cudaFree(d_stack);
    cudaFree(d_buffer);
    cudaFree(d_sum);

    return 0;
}
