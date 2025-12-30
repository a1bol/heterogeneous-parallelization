#include <iostream>  // Библиотека ввода-вывода
#include <vector>    // Контейнер vector
#include <algorithm> // Алгоритмы (функция swap)
#include <chrono>    // Измерение времени
#include <cstdlib>   // Стандартная библиотека (rand, srand)
#include <ctime>     // Функции времени
#include <omp.h>     // OpenMP для параллелизации

using namespace std;
using namespace std::chrono;

// Задание 3: Сортировка выбором с OpenMP
// Цель: Реализовать и сравнить последовательную и параллельную сортировку выбором

// --- ПОСЛЕДОВАТЕЛЬНАЯ СОРТИРОВКА ВЫБОРОМ ---
// Сортировка выбором работает путем многократного поиска минимального элемента
// из неотсортированной части и помещения его в начало
void sequentialSelectionSort(vector<int>& arr) {
    int n = arr.size();
    
    // Внешний цикл: перемещает границу отсортированной части
    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i; // Предполагаем, что текущий элемент - минимальный
        
        // Внутренний цикл: находим реальный минимум в оставшейся неотсортированной части
        for (int j = i + 1; j < n; ++j) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j; // Обновляем индекс минимума, если нашли меньший элемент
            }
        }
        
        // Меняем местами найденный минимальный элемент с элементом в позиции i
        swap(arr[min_idx], arr[i]);
    }
}

// --- ПАРАЛЛЕЛЬНАЯ СОРТИРОВКА ВЫБОРОМ (OpenMP) ---
// Основная сложность: сортировка выбором имеет присущие последовательные зависимости
// Внешний цикл ДОЛЖЕН быть последовательным (каждая итерация зависит от предыдущих)
// Но мы можем распараллелить внутренний цикл (поиск минимума)
void parallelSelectionSort(vector<int>& arr) {
    int n = arr.size();
    
    // Внешний цикл НЕ МОЖЕТ быть распараллелен - он должен быть последовательным
    // Каждая итерация i помещает i-й наименьший элемент в позицию i
    // Итерация i+1 зависит от полного завершения итерации i
    for (int i = 0; i < n - 1; ++i) {
        
        int min_idx = i;      // Глобальный индекс минимума
        int min_val = arr[i]; // Глобальное значение минимума
         
        // Параллельная область для поиска минимума
        #pragma omp parallel
        {
            int local_min_idx = min_idx; // Локальный индекс минимума потока
            int local_min_val = min_val; // Локальное значение минимума потока
            
            // Распараллеливаем внутренний цикл - поиск минимума параллельно
            // Каждый поток ищет в своей части массива
            // nowait: не синхронизироваться в конце цикла for
            // (вместо этого синхронизация в критической секции)
            #pragma omp for nowait
            for (int j = i + 1; j < n; ++j) {
                // Каждый поток находит минимум в своем диапазоне
                if (arr[j] < local_min_val) {
                    local_min_val = arr[j];
                    local_min_idx = j;
                }
            }
            
            // Критическая секция: объединяем результаты всех потоков
            // Только один поток выполняет это одновременно
            #pragma omp critical
            {
                // Сравниваем локальный минимум этого потока с глобальным минимумом
                if (local_min_val < min_val) {
                    min_val = local_min_val;
                    min_idx = local_min_idx;
                }
            }
        }
        
        // После параллельной области у нас есть глобальный минимум
        // Меняем его местами с текущей позицией
        swap(arr[i], arr[min_idx]);
    }
}

// Вспомогательная функция для проверки, отсортирован ли массив
bool isSorted(const vector<int>& arr) {
    for (size_t i = 0; i < arr.size() - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            return false; // Нашли элемент больше следующего
        }
    }
    return true; // Все элементы в порядке
}

// Вспомогательная функция для измерения производительности сортировки
void measureSortPerformance(void (*sortFunc)(vector<int>&), vector<int> arr, const string& name) {
    // Выводим название алгоритма
    cout << "\n" << name << endl;
    cout << "Array size: " << arr.size() << endl;
    
    // Начинаем отсчет времени
    auto start = high_resolution_clock::now();
    
    // Выполняем функцию сортировки
    sortFunc(arr);
    
    // Останавливаем отсчет времени
    auto stop = high_resolution_clock::now();
    
    // Вычисляем длительность в миллисекундах
    auto duration = duration_cast<milliseconds>(stop - start);
    
    // Проверяем корректность
    if (isSorted(arr)) {
        cout << "✓ Correctly sorted" << endl;
    } else {
        cout << "✗ Sorting failed!" << endl;
    }
    
    // Выводим время выполнения
    cout << "Time: " << duration.count() << " ms" << endl;
}

int main() {
    // Инициализация генератора случайных чисел
    srand(time(0));

    // Тестируем с двумя разными размерами массива
    vector<int> sizes = {1000, 10000};
    
    cout << "=== Task 3: Selection Sort with OpenMP ===" << endl;
    cout << "Testing sequential vs parallel implementations\n" << endl;

    // Тестируем каждый размер массива
    for (int n : sizes) {
        cout << "\n========================================" << endl;
        cout << "Testing with array size: " << n << endl;
        cout << "========================================" << endl;
        
        // Создаем и заполняем массив случайными числами
        vector<int> data(n);
        for (int i = 0; i < n; ++i) {
            data[i] = rand() % 100000; // Случайные числа от 0 до 99999
        }

        // Тестируем последовательную версию
        measureSortPerformance(sequentialSelectionSort, data, "Sequential Selection Sort");
        
        // Тестируем параллельную версию
        measureSortPerformance(parallelSelectionSort, data, "Parallel Selection Sort (OpenMP)");
        
        // Сравниваем производительность
        // Примечание: Для сортировки выбором параллельная версия может быть не всегда быстрее
        // потому что алгоритм имеет ограниченные возможности параллелизации
        // (только внутренний цикл можно распараллелить)
    }
    
    cout << "\n========================================" << endl;
    cout << "Analysis:" << endl;
    cout << "========================================" << endl;
    cout << "Selection sort has limited parallelization potential because:" << endl;
    cout << "1. The outer loop must remain sequential (dependencies between iterations)" << endl;
    cout << "2. Only the min-finding step (inner loop) can be parallelized" << endl;
    cout << "3. For small arrays, parallel overhead may exceed benefits" << endl;
    cout << "4. Better parallelizable sorts exist (e.g., merge sort, quick sort)" << endl;

    return 0;
}
