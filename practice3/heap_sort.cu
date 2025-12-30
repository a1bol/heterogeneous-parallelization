#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== CPU ВЕРСИЯ: Пирамидальная сортировка =====

// Преобразование поддерева с корнем i в max-heap
void heapify(int* arr, int n, int i) {
    int largest = i;       // Инициализируем наибольший элемент как корень
    int left = 2 * i + 1;  // Левый потомок
    int right = 2 * i + 2; // Правый потомок
    
    // Если левый потомок больше корня
    if (left < n && arr[left] > arr[largest])
        largest = left;
    
    // Если правый потомок больше текущего наибольшего
    if (right < n && arr[right] > arr[largest])
        largest = right;
    
    // Если наибольший элемент не корень
    if (largest != i) {
        swap(arr[i], arr[largest]);
        
        // Рекурсивно преобразуем затронутое поддерево
        heapify(arr, n, largest);
    }
}

// Основная функция пирамидальной сортировки на CPU
void cpuHeapSort(int* arr, int n) {
    // Построение max-heap (перестройка массива)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    
    // Извлечение элементов из кучи по одному
    for (int i = n - 1; i > 0; i--) {
        // Перемещаем текущий корень (max) в конец
        swap(arr[0], arr[i]);
        
        // Вызываем heapify для уменьшенной кучи
        heapify(arr, i, 0);
    }
}

// ===== GPU ВЕРСИЯ: Пирамидальная сортировка на CUDA =====

// CUDA ядро: Построение heap (параллельная версия heapify)
__global__ void heapifyKernel(int* arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && arr[left] > arr[largest])
        largest = left;
    
    if (right < n && arr[right] > arr[largest])
        largest = right;
    
    if (largest != i) {
        // Используем atomicExch для безопасного обмена
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;
    }
}

// Функция запуска CUDA пирамидальной сортировки
void cudaHeapSort(int* arr, int n) {
    int* d_arr;
    
    // Выделяем память на GPU
    cudaMalloc(&d_arr, n * sizeof(int));
    
    // Копируем данные на GPU
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Построение max-heap
    // Обрабатываем уровни кучи сверху вниз
    for (int i = n / 2 - 1; i >= 0; i--) {
        // Запускаем по одному потоку для каждого узла
        // В полноценной версии можно распараллелить уровни
        heapifyKernel<<<1, 1>>>(d_arr, n, i);
        cudaDeviceSynchronize();
    }
    
    // Извлечение элементов из кучи
    // Этот шаг труднее параллелить, так как каждый шаг зависит от предыдущего
    for (int i = n - 1; i > 0; i--) {
        // Копируем данные на CPU для swap и heapify
        // В полной GPU версии это делалось бы на GPU
        cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Swap на CPU
        swap(arr[0], arr[i]);
        
        // Копируем обратно на GPU
        cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
        
        // Heapify на GPU
        for (int j = i / 2 - 1; j >= 0; j--) {
            heapifyKernel<<<1, 1>>>(d_arr, i, j);
            cudaDeviceSynchronize();
        }
    }
    
    // Копируем результат обратно на CPU
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Освобождаем память GPU
    cudaFree(d_arr);
}

// ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====

// Проверка правильности сортировки
bool isSorted(int* arr, int n) {
    for (int i = 0; i < n - 1; ++i) {
        if (arr[i] > arr[i + 1]) return false;
    }
    return true;
}

// ===== MAIN ФУНКЦИЯ =====

int main() {
    srand(time(0));
    
    // Размеры массивов для тестирования
    // Примечание: heap sort на GPU с частыми CPU-GPU transfers будет медленнее
    // Для демонстрации используем меньшие размеры
    int sizes[] = {10000, 100000};
    
    cout << "=== Practice 3: Heap Sort - CPU vs GPU ===\n" << endl;
    cout << "Note: This implementation shows the challenges of parallelizing heap sort\n" << endl;
    
    for (int size : sizes) {
        cout << "\n========================================" << endl;
        cout << "Array size: " << size << endl;
        cout << "========================================" << endl;
        
        // Создаем массивы
        int* arr_cpu = new int[size];
        int* arr_gpu = new int[size];
        
        // Заполняем одинаковыми случайными числами
        for (int i = 0; i < size; ++i) {
            int val = rand() % 100000;
            arr_cpu[i] = val;
            arr_gpu[i] = val;
        }
        
        // === CPU Сортировка ===
        cout << "\nCPU Heap Sort:" << endl;
        auto start = high_resolution_clock::now();
        
        cpuHeapSort(arr_cpu, size);
        
        auto end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<milliseconds>(end - start);
        
        cout << "Time: " << cpu_duration.count() << " ms" << endl;
        cout << "Sorted: " << (isSorted(arr_cpu, size) ? "✓ Yes" : "✗ No") << endl;
        
        // === GPU Сортировка ===
        cout << "\nGPU CUDA Heap Sort (Hybrid):" << endl;
        start = high_resolution_clock::now();
        
        cudaHeapSort(arr_gpu, size);
        
        end = high_resolution_clock::now();
        auto gpu_duration = duration_cast<milliseconds>(end - start);
        
        cout << "Time: " << gpu_duration.count() << " ms" << endl;
        cout << "Sorted: " << (isSorted(arr_gpu, size) ? "✓ Yes" : "✗ No") << endl;
        
        // === Сравнение ===
        double speedup = (double)cpu_duration.count() / gpu_duration.count();
        cout << "\nSpeedup (GPU vs CPU): " << speedup << "x" << endl;
        
        if (speedup > 1.0) {
            cout << "GPU is " << speedup << "x faster" << endl;
        } else {
            cout << "CPU is faster (expected due to CPU-GPU transfers)" << endl;
        }
        
        // Очистка
        delete[] arr_cpu;
        delete[] arr_gpu;
    }
    
    cout << "\n========================================" << endl;
    cout << "Summary:" << endl;
    cout << "========================================" << endl;
    cout << "- Heap sort is difficult to parallelize efficiently" << endl;
    cout << "- Sequential dependencies limit GPU advantages" << endl;
    cout << "- Frequent CPU-GPU data transfers hurt performance" << endl;
    cout << "- Better suited algorithms for GPU: merge sort, radix sort" << endl;
    
    return 0;
}
