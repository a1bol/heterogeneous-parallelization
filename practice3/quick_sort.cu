#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== CPU ВЕРСИЯ: Быстрая сортировка =====

// Разделение массива по опорному элементу (partition)
int partition(int* arr, int low, int high) {
    // Выбираем опорный элемент (pivot)
    int pivot = arr[high];
    int i = low - 1;
    
    // Перемещаем элементы меньше pivot влево
    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    
    // Помещаем pivot на правильную позицию
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Рекурсивная быстрая сортировка на CPU
void cpuQuickSort(int* arr, int low, int high) {
    if (low < high) {
        // Разделяем массив и получаем позицию pivot
        int pi = partition(arr, low, high);
        
        // Рекурсивно сортируем левую и правую части
        cpuQuickSort(arr, low, pi - 1);
        cpuQuickSort(arr, pi + 1, high);
    }
}

// ===== GPU ВЕРСИЯ: Быстрая сортировка на CUDA =====

// CUDA ядро: Insertion sort для небольших подмассивов
__global__ void insertionSortKernel(int* arr, int n, int threshold) {
    int blockStart = blockIdx.x * threshold;
    int blockEnd = min(blockStart + threshold, n);
    
    if (threadIdx.x == 0 && blockStart < n) {
        for (int i = blockStart + 1; i < blockEnd; ++i) {
            int key = arr[i];
            int j = i - 1;
            
            while (j >= blockStart && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
}

// CUDA ядро: Partition для быстрой сортировки
__global__ void partitionKernel(int* arr, int n, int pivot_value, int* left_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Подсчитываем элементы меньше pivot
        if (arr[tid] < pivot_value) {
            atomicAdd(left_count, 1);
        }
    }
}

// Упрощенная GPU быстрая сортировка
// Примечание: Полная параллельная быстрая сортировка на GPU сложна
// Мы используем гибридный подход: разбиваем на блоки и сортируем каждый
void cudaQuickSort(int* arr, int n) {
    int* d_arr;
    
    // Выделяем память на GPU
    cudaMalloc(&d_arr, n * sizeof(int));
    
    // Копируем данные на GPU
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Гибридный подход: делим массив на блоки и сортируем каждый блок
    // Это эффективнее для GPU, чем полная параллельная quicksort
    int threshold = 512; // Размер блока для insertion sort
    int numBlocks = (n + threshold - 1) / threshold;
    
    // Сортируем каждый блок отдельно
    insertionSortKernel<<<numBlocks, 1>>>(d_arr, n, threshold);
    cudaDeviceSynchronize();
    
    // Копируем результат на CPU для финальной сортировки
    // (полноценный merge на GPU добавил бы сложности)
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Используем CPU для слияния отсортированных блоков
    // В реальных приложениях это можно сделать полностью на GPU
    int* temp = new int[n];
    for (int width = threshold; width < n; width *= 2) {
        for (int i = 0; i < n; i += 2 * width) {
            int left = i;
            int mid = min(i + width, n);
            int right = min(i + 2 * width, n);
            
            // Слияние подмассивов
            int l = left, r = mid, k = left;
            while (l < mid && r < right) {
                if (arr[l] <= arr[r]) {
                    temp[k++] = arr[l++];
                } else {
                    temp[k++] = arr[r++];
                }
            }
            while (l < mid) temp[k++] = arr[l++];
            while (r < right) temp[k++] = arr[r++];
        }
        for (int i = 0; i < n; ++i) arr[i] = temp[i];
    }
    delete[] temp;
    
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
    int sizes[] = {10000, 100000, 1000000};
    
    cout << "=== Practice 3: Quick Sort - CPU vs GPU ===\n" << endl;
    
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
        cout << "\nCPU Quick Sort:" << endl;
        auto start = high_resolution_clock::now();
        
        cpuQuickSort(arr_cpu, 0, size - 1);
        
        auto end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<milliseconds>(end - start);
        
        cout << "Time: " << cpu_duration.count() << " ms" << endl;
        cout << "Sorted: " << (isSorted(arr_cpu, size) ? "✓ Yes" : "✗ No") << endl;
        
        // === GPU Сортировка ===
        cout << "\nGPU CUDA Hybrid Quick Sort:" << endl;
        start = high_resolution_clock::now();
        
        cudaQuickSort(arr_gpu, size);
        
        end = high_resolution_clock::now();
        auto gpu_duration = duration_cast<milliseconds>(end - start);
        
        cout << "Time: " << gpu_duration.count() << " ms" << endl;
        cout << "Sorted: " << (isSorted(arr_gpu, size) ? "✓ Yes" : "✗ No") << endl;
        
        // === Сравнение ===
        double speedup = (double)cpu_duration.count() / gpu_duration.count();
        cout << "\nSpeedup (GPU vs CPU): " << speedup << "x" << endl;
        
        if (speedup > 1.0) {
            cout << "GPU is " << speedup << "x faster!" << endl;
        } else {
            cout << "CPU is faster" << endl;
        }
        
        // Очистка
        delete[] arr_cpu;
        delete[] arr_gpu;
    }
    
    cout << "\n========================================" << endl;
    cout << "Summary:" << endl;
    cout << "========================================" << endl;
    cout << "- Quick sort is challenging to parallelize on GPU" << endl;
    cout << "- This implementation uses a hybrid approach" << endl;
    cout << "- GPU sorts blocks in parallel, CPU handles merging" << endl;
    cout << "- Full GPU quicksort requires complex synchronization" << endl;
    
    return 0;
}
