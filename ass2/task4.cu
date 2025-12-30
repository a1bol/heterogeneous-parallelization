#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Задание 4: Сортировка слиянием на GPU с использованием CUDA
// Цель: Реализовать параллельную сортировку слиянием используя CUDA

// CUDA ядро для слияния двух отсортированных подмассивов
// Это выполняется на GPU - каждый поток обрабатывает одну операцию слияния
__global__ void mergeSortedSubarrays(int* arr, int* temp, int width, int n) {
    // Вычисляем начальную позицию для операции слияния этого потока
    // Каждый поток сливает два смежных отсортированных подмассива размера 'width'
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = 2 * tid * width;
    
    // Проверяем, есть ли у этого потока работа
    if (start >= n) return;
    
    // Вычисляем границы для двух подмассивов, которые нужно слить
    int mid = min(start + width, n);     // Конец первого подмассива
    int end = min(start + 2 * width, n); // Конец второго подмассива
    
    // Указатели для слияния
    int i = start;  // Указатель на первый подмассив
    int j = mid;    // Указатель на второй подмассив
    int k = start;  // Указатель на позицию вывода
    
    // Сливаем два отсортированных подмассива в временный массив
    // Аналогично классическому алгоритму слияния
    while (i < mid && j < end) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    // Копируем оставшиеся элементы из первого подмассива (если есть)
    while (i < mid) {
        temp[k++] = arr[i++];
    }
    
    // Копируем оставшиеся элементы из второго подмассива (если есть)
    while (j < end) {
        temp[k++] = arr[j++];
    }
}

// CUDA ядро для сортировки отдельных блоков
// Каждый блок сортирует свою назначенную часть используя сортировку вставками
// (Простая и эффективная для небольших массивов)
__global__ void insertionSortBlocks(int* arr, int n, int blockSize) {
    // Вычисляем, какую часть массива должен сортировать этот блок
    int blockStart = blockIdx.x * blockSize;
    int blockEnd = min(blockStart + blockSize, n);
    
    // Сортировка вставками для части этого блока
    // Только поток 0 в каждом блоке выполняет сортировку
    // (Можно дополнительно распараллелить, но сортировка вставками по своей природе последовательна)
    if (threadIdx.x == 0) {
        for (int i = blockStart + 1; i < blockEnd; ++i) {
            int key = arr[i];
            int j = i - 1;
            
            // Сдвигаем элементы больше key вправо
            while (j >= blockStart && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
}

// Хост-функция для выполнения сортировки слиянием на GPU
void cudaMergeSort(int* arr, int n) {
    // Указатели на устройство (GPU)
    int* d_arr;      // Массив на GPU
    int* d_temp;     // Временный массив для слияния
    
    // Выделение памяти на GPU
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));
    
    // Копирование данных с хоста (CPU) на устройство (GPU)
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Шаг 1: Сортировка отдельных блоков используя сортировку вставками
    // Каждый блок сортирует небольшую часть массива
    int threadsPerBlock = 256;
    int blockSize = 256; // Размер каждого блока для сортировки
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    insertionSortBlocks<<<numBlocks, 1>>>(d_arr, n, blockSize);
    cudaDeviceSynchronize(); // Ждем завершения работы GPU
    
    // Шаг 2: Итеративное слияние отсортированных блоков
    // На каждой итерации сливаем пары отсортированных подмассивов
    // Width начинается с blockSize и удваивается на каждой итерации
    for (int width = blockSize; width < n; width *= 2) {
        // Вычисляем, сколько операций слияния нам нужно
        int numMerges = (n + 2 * width - 1) / (2 * width);
        
        // Запускаем ядро для выполнения слияний
        int threads = min(256, numMerges);
        int blocks = (numMerges + threads - 1) / threads;
        
        mergeSortedSubarrays<<<blocks, threads>>>(d_arr, d_temp, width, n);
        cudaDeviceSynchronize(); // Ждем завершения слияний
        
        // Меняем местами arr и temp для следующей итерации
        // Это эффективно - мы просто меняем указатели, а не копируем данные
        int* tmp = d_arr;
        d_arr = d_temp;
        d_temp = tmp;
    }
    
    // Копируем отсортированные данные обратно с GPU на CPU
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Освобождаем память GPU
    cudaFree(d_arr);
    cudaFree(d_temp);
}

// Последовательная сортировка слиянием для сравнения
void sequentialMergeSort(int* arr, int* temp, int left, int right) {
    if (left >= right) return;
    
    int mid = left + (right - left) / 2;
    
    // Рекурсивно сортируем левую и правую половины
    sequentialMergeSort(arr, temp, left, mid);
    sequentialMergeSort(arr, temp, mid + 1, right);
    
    // Сливаем отсортированные половины
    int i = left, j = mid + 1, k = left;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    // Копируем результат слияния обратно в исходный массив
    for (i = left; i <= right; ++i) {
        arr[i] = temp[i];
    }
}

// Вспомогательная функция для проверки, отсортирован ли массив
bool isSorted(int* arr, int n) {
    for (int i = 0; i < n - 1; ++i) {
        if (arr[i] > arr[i + 1]) return false;
    }
    return true;
}

int main() {
    srand(time(0));
    
    // Тестируем с двумя разными размерами массива
    int sizes[] = {10000, 100000};
    
    cout << "=== Task 4: Merge Sort on GPU using CUDA ===" << endl;
    cout << "Comparing CPU (sequential) vs GPU (CUDA) implementations\n" << endl;
    
    for (int size : sizes) {
        cout << "\n========================================" << endl;
        cout << "Array size: " << size << endl;
        cout << "========================================" << endl;
        
        // Выделяем и инициализируем массивы
        int* arr_cpu = new int[size];
        int* arr_gpu = new int[size];
        int* temp = new int[size];
        
        // Заполняем случайными числами
        for (int i = 0; i < size; ++i) {
            int val = rand() % 100000;
            arr_cpu[i] = val;
            arr_gpu[i] = val; // Те же данные для честного сравнения
        }
        
        // --- CPU Последовательная сортировка слиянием ---
        cout << "\nCPU Sequential Merge Sort:" << endl;
        auto start = high_resolution_clock::now();
        
        sequentialMergeSort(arr_cpu, temp, 0, size - 1);
        
        auto end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<milliseconds>(end - start);
        
        cout << "Time: " << cpu_duration.count() << " ms" << endl;
        cout << "Sorted: " << (isSorted(arr_cpu, size) ? "✓ Yes" : "✗ No") << endl;
        
        // --- GPU CUDA Сортировка слиянием ---
        cout << "\nGPU CUDA Merge Sort:" << endl;
        start = high_resolution_clock::now();
        
        cudaMergeSort(arr_gpu, size);
        
        end = high_resolution_clock::now();
        auto gpu_duration = duration_cast<milliseconds>(end - start);
        
        cout << "Time: " << gpu_duration.count() << " ms" << endl;
        cout << "Sorted: " << (isSorted(arr_gpu, size) ? "✓ Yes" : "✗ No") << endl;
        
        // --- Сравнение производительности ---
        double speedup = (double)cpu_duration.count() / gpu_duration.count();
        cout << "\nSpeedup (GPU vs CPU): " << speedup << "x" << endl;
        
        if (speedup > 1.0) {
            cout << "GPU is faster!" << endl;
        } else {
            cout << "CPU is faster (GPU overhead > benefits for this size)" << endl;
        }
        
        // Очистка
        delete[] arr_cpu;
        delete[] arr_gpu;
        delete[] temp;
    }
    
    cout << "\n========================================" << endl;
    cout << "Notes:" << endl;
    cout << "========================================" << endl;
    cout << "1. GPU performance includes memory transfer overhead (CPU↔GPU)" << endl;
    cout << "2. Larger arrays typically show better GPU speedup" << endl;
    cout << "3. Block size and grid configuration affect GPU performance" << endl;
    cout << "4. Modern GPUs excel at data-parallel tasks like sorting" << endl;
    
    return 0;
}
