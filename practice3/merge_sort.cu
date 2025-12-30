#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== CPU ВЕРСИЯ: Сортировка слиянием =====

// Слияние двух отсортированных подмассивов
void merge(int* arr, int* temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    
    // Сливаем элементы из двух половин
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    // Копируем оставшиеся элементы из левой половины
    while (i <= mid) temp[k++] = arr[i++];
    // Копируем оставшиеся элементы из правой половины
    while (j <= right) temp[k++] = arr[j++];
    
    // Копируем отсортированные элементы обратно в исходный массив
    for (i = left; i <= right; ++i) {
        arr[i] = temp[i];
    }
}

// Рекурсивная сортировка слиянием на CPU
void cpuMergeSort(int* arr, int* temp, int left, int right) {
    if (left >= right) return;
    
    int mid = left + (right - left) / 2;
    
    // Рекурсивно сортируем левую и правую половины
    cpuMergeSort(arr, temp, left, mid);
    cpuMergeSort(arr, temp, mid + 1, right);
    
    // Сливаем отсортированные половины
    merge(arr, temp, left, mid, right);
}

// ===== GPU ВЕРСИЯ: Сортировка слиянием на CUDA =====

// CUDA ядро: Сортировка небольших блоков используя insertion sort
__global__ void insertionSortKernel(int* arr, int n, int blockSize) {
    // Вычисляем начало и конец блока для этого CUDA блока
    int blockStart = blockIdx.x * blockSize;
    int blockEnd = min(blockStart + blockSize, n);
    
    // Только первый поток в каждом блоке выполняет сортировку
    if (threadIdx.x == 0) {
        for (int i = blockStart + 1; i < blockEnd; ++i) {
            int key = arr[i];
            int j = i - 1;
            
            // Сдвигаем элементы, которые больше key
            while (j >= blockStart && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
}

// CUDA ядро: Слияние отсортированных подмассивов
__global__ void mergeKernel(int* arr, int* temp, int width, int n) {
    // Каждый поток обрабатывает одно слияние
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = 2 * tid * width;
    
    if (start >= n) return;
    
    // Определяем границы двух подмассивов для слияния
    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);
    
    int i = start;
    int j = mid;
    int k = start;
    
    // Сливаем два отсортированных подмассива
    while (i < mid && j < end) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    // Копируем оставшиеся элементы
    while (i < mid) temp[k++] = arr[i++];
    while (j < end) temp[k++] = arr[j++];
}

// Функция запуска CUDA сортировки слиянием
void cudaMergeSort(int* arr, int n) {
    int* d_arr;
    int* d_temp;
    
    // Выделяем память на GPU
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));
    
    // Копируем данные на GPU
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Шаг 1: Сортируем блоки по 256 элементов
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    insertionSortKernel<<<numBlocks, 1>>>(d_arr, n, blockSize);
    cudaDeviceSynchronize();
    
    // Шаг 2: Итеративно сливаем отсортированные блоки
    for (int width = blockSize; width < n; width *= 2) {
        int numMerges = (n + 2 * width - 1) / (2 * width);
        int threads = min(256, numMerges);
        int blocks = (numMerges + threads - 1) / threads;
        
        mergeKernel<<<blocks, threads>>>(d_arr, d_temp, width, n);
        cudaDeviceSynchronize();
        
        // Меняем указатели местами
        int* tmp = d_arr;
        d_arr = d_temp;
        d_temp = tmp;
    }
    
    // Копируем результат обратно на CPU
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Освобождаем память GPU
    cudaFree(d_arr);
    cudaFree(d_temp);
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
    
    cout << "=== Practice 3: Merge Sort - CPU vs GPU ===\n" << endl;
    
    for (int size : sizes) {
        cout << "\n========================================" << endl;
        cout << "Array size: " << size << endl;
        cout << "========================================" << endl;
        
        // Создаем массивы
        int* arr_cpu = new int[size];
        int* arr_gpu = new int[size];
        int* temp = new int[size];
        
        // Заполняем одинаковыми случайными числами
        for (int i = 0; i < size; ++i) {
            int val = rand() % 100000;
            arr_cpu[i] = val;
            arr_gpu[i] = val;
        }
        
        // === CPU Сортировка ===
        cout << "\nCPU Merge Sort:" << endl;
        auto start = high_resolution_clock::now();
        
        cpuMergeSort(arr_cpu, temp, 0, size - 1);
        
        auto end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<milliseconds>(end - start);
        
        cout << "Time: " << cpu_duration.count() << " ms" << endl;
        cout << "Sorted: " << (isSorted(arr_cpu, size) ? "✓ Yes" : "✗ No") << endl;
        
        // === GPU Сортировка ===
        cout << "\nGPU CUDA Merge Sort:" << endl;
        start = high_resolution_clock::now();
        
        cudaMergeSort(arr_gpu, size);
        
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
            cout << "CPU is faster (GPU overhead dominates)" << endl;
        }
        
        // Очистка
        delete[] arr_cpu;
        delete[] arr_gpu;
        delete[] temp;
    }
    
    cout << "\n========================================" << endl;
    cout << "Summary:" << endl;
    cout << "========================================" << endl;
    cout << "- Merge sort is well-suited for parallelization" << endl;
    cout << "- GPU shows better speedup on larger arrays" << endl;
    cout << "- Memory transfer overhead affects small arrays" << endl;
    
    return 0;
}
