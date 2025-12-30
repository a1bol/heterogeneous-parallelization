#include <iostream>  // Библиотека ввода-вывода (cout, cin)
#include <vector>    // Контейнер vector (динамический массив)
#include <algorithm> // Библиотека алгоритмов
#include <cstdlib>   // Стандартная библиотека (rand, srand)
#include <ctime>     // Библиотека времени (функция time для инициализации генератора)
#include <chrono>    // Высокоточное измерение времени
#include <limits>    // Численные лимиты (для макс/мин значений int)
#include <omp.h>     // Библиотека OpenMP для параллельных вычислений

using namespace std; // Использование стандартного пространства имен

// Задание 2: Поиск минимального и максимального значений в массиве
// Цель: Сравнить последовательную и параллельную (OpenMP) реализации поиска мин/макс

// --- ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ---
// Эта функция находит мин и макс значения используя простой последовательный цикл
void sequentialMinMax(const vector<int>& arr, int& minVal, int& maxVal) {
    // Инициализируем minVal максимально возможным значением int
    // Инициализируем maxVal минимально возможным значением int
    minVal = numeric_limits<int>::max();
    maxVal = numeric_limits<int>::min();
    
    // Проходим по всем элементам массива
    // Range-based for loop удобен для векторов
    for (int val : arr) {
        if (val < minVal) minVal = val; // Обновляем минимум, если нашли меньшее значение
        if (val > maxVal) maxVal = val; // Обновляем максимум, если нашли большее значение
    }
}

// --- ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ (OpenMP) ---
// Эта функция использует OpenMP для распараллеливания поиска мин/макс
void parallelMinMax(const vector<int>& arr, int& minVal, int& maxVal) {
    // Инициализируем глобальные мин/макс значения
    minVal = numeric_limits<int>::max();
    maxVal = numeric_limits<int>::min();

    // Создаем параллельную область
    // #pragma omp parallel - создает команду потоков
    #pragma omp parallel
    {
        // Каждый поток поддерживает свои локальные мин/макс
        // Это избегает состояния гонки при обновлении общих переменных
        int localMin = numeric_limits<int>::max();
        int localMax = numeric_limits<int>::min();

        // Распределяем итерации цикла между потоками
        // #pragma omp for - автоматически делит диапазон цикла между потоками
        #pragma omp for
        for (size_t i = 0; i < arr.size(); ++i) {
            // Каждый поток обрабатывает свою часть массива
            if (arr[i] < localMin) localMin = arr[i];
            if (arr[i] > localMax) localMax = arr[i];
        }

        // После цикла каждый поток нашел мин/макс в своей части
        // Теперь нужно безопасно объединить эти результаты
        
        // #pragma omp critical - Критическая секция
        // Только ОДИН поток может выполнять этот код одновременно
        // Это предотвращает гонки данных при обновлении глобальных мин/макс
        #pragma omp critical
        {
            if (localMin < minVal) minVal = localMin;
            if (localMax > maxVal) maxVal = localMax;
        }
    }
}

int main() {
    // Инициализация генератора случайных чисел
    // Используем текущее время как seed, чтобы числа были разными при каждом запуске
    srand(static_cast<unsigned int>(time(0)));
    
    const int SIZE = 10000; // Размер массива: 10,000 элементов
    cout << "Generating array of " << SIZE << " random integers..." << endl;
    
    // Создаем вектор размера SIZE
    vector<int> arr(SIZE);
    
    // Заполняем массив случайными числами
    // auto& val позволяет нам напрямую изменять элементы вектора
    for (auto& val : arr) {
        val = rand(); // Случайное целое число (обычно от 0 до RAND_MAX)
    }

    int seqMin, seqMax; // Переменные для результатов последовательной версии
    int parMin, parMax; // Переменные для результатов параллельной версии

    // --- Последовательная реализация ---
    cout << "\n=== Sequential Version ===" << endl;
    
    // Начинаем отсчет времени
    auto start = chrono::high_resolution_clock::now();
    
    sequentialMinMax(arr, seqMin, seqMax);
    
    // Останавливаем отсчет времени
    auto end = chrono::high_resolution_clock::now();
    
    // Вычисляем длительность в секундах
    chrono::duration<double> seqDuration = end - start;

    cout << "Min: " << seqMin << ", Max: " << seqMax << endl;
    cout << "Time: " << seqDuration.count() << " seconds" << endl;

    // --- Параллельная реализация (OpenMP) ---
    cout << "\n=== Parallel Version (OpenMP) ===" << endl;
    
    start = chrono::high_resolution_clock::now();
    
    parallelMinMax(arr, parMin, parMax);
    
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> parDuration = end - start;

    cout << "Min: " << parMin << ", Max: " << parMax << endl;
    cout << "Time: " << parDuration.count() << " seconds" << endl;
    
    // --- Сравнение производительности ---
    cout << "\n=== Performance Analysis ===" << endl;
    
    // Проверяем, что оба метода нашли одинаковые результаты
    if (seqMin == parMin && seqMax == parMax) {
        cout << "✓ Results match: Both methods found the same min/max values" << endl;
    } else {
        cout << "✗ Error: Results do not match!" << endl;
    }
    
    // Вычисляем ускорение (во сколько раз параллельная версия быстрее)
    double speedup = seqDuration.count() / parDuration.count();
    cout << "Ускорение: " << speedup << "x" << endl;
    
    // Даем интерпретацию
    cout << "\nConclusion: ";
    if (speedup > 1.0) {
        cout << "Parallel version is faster by " << speedup << "x" << endl;
    } else if (speedup < 1.0) {
        cout << "Sequential version is faster (parallel overhead exceeds benefits)" << endl;
    } else {
        cout << "Both versions perform similarly" << endl;
    }

    return 0;
}
