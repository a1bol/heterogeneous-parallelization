#include <iostream>  // Подключение библиотеки ввода-вывода
#include <vector>    // Подключение контейнера vector (удобная альтернатива динамическим массивам)
#include <algorithm> // Подключение алгоритмов
#include <cstdlib>   // Подключение для rand, srand
#include <ctime>     // Подключение для time
#include <chrono>    // Подключение для точного измерения времени выполнения
#include <limits>    // Подключение для std::numeric_limits (получение макс./мин. возможных значений int)
#include <omp.h>     // Подключение библиотеки OpenMP для параллельных вычислений

using namespace std; // Использование стандартного пространства имен

// Задания 2 и 3: Поиск минимального и максимального элементов в массиве (1 млн элементов)
// Сравнение производительности последовательного и параллельного алгоритмов

// --- ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ---
void sequentialMinMax(const vector<int>& arr, int& minVal, int& maxVal) {
    // Инициализируем minVal максимально возможным int, а maxVal - минимально возможным
    minVal = numeric_limits<int>::max();
    maxVal = numeric_limits<int>::min(); // Или просто arr[0], если массив не пуст
    
    // Проходим по всем элементам вектора (range-based for loop)
    for (int val : arr) {
        if (val < minVal) minVal = val; // Нашли новое минимальное
        if (val > maxVal) maxVal = val; // Нашли новое максимальное
    }
}

// --- ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ (OpenMP) ---
void parallelMinMax(const vector<int>& arr, int& minVal, int& maxVal) {
    minVal = numeric_limits<int>::max();
    maxVal = numeric_limits<int>::min();

    // Создаем параллельную область
    // #pragma omp parallel - создает команду потоков
    #pragma omp parallel
    {
        // Локальные переменные для каждого потока
        // Каждый поток будет искать мин/макс только в СВОЕЙ части массива
        int localMin = numeric_limits<int>::max();
        int localMax = numeric_limits<int>::min();

        // Распределяем итерации цикла между потоками
        // #pragma omp for - автоматически делит диапазон i на части для каждого потока
        #pragma omp for
        for (size_t i = 0; i < arr.size(); ++i) {
            if (arr[i] < localMin) localMin = arr[i];
            if (arr[i] > localMax) localMax = arr[i];
        }

        // После завершения цикла каждый поток имеет свои localMin и localMax
        // Теперь нужно безопасно обновить глобальные minVal и maxVal
        
        // #pragma omp critical - Критическая секция
        // Код внутри выполняется только ОДНИМ потоком одновременно
        // Это предотвращает "гонки данных" (data race) при записи в общие переменные
        #pragma omp critical
        {
            if (localMin < minVal) minVal = localMin;
            if (localMax > maxVal) maxVal = localMax;
        }
    }
}

int main() {
    // Инициализация рандома
    srand(static_cast<unsigned int>(time(0)));
    
    const int SIZE = 1000000; // Размер массива: 1 миллион элементов
    cout << "Генерация массива из " << SIZE << " целых чисел..." << endl;
    
    // Создаем вектор размера SIZE
    vector<int> arr(SIZE);
    
    // Заполняем случайными числами
    // auto& val позволяет изменять элементы вектора напрямую
    for (auto& val : arr) {
        val = rand(); // Случайное целое число (обычно от 0 до RAND_MAX)
    }

    int seqMin, seqMax;
    int parMin, parMax;

    // --- Задание 2: Последовательное измерение ---
    cout << "\nЗапуск последовательного поиска..." << endl;
    
    // Засекаем время начала
    auto start = chrono::high_resolution_clock::now();
    
    sequentialMinMax(arr, seqMin, seqMax);
    
    // Засекаем время окончания
    auto end = chrono::high_resolution_clock::now();
    
    // Вычисляем длительность в секундах (double)
    chrono::duration<double> seqDuration = end - start;

    cout << "Последовательные результаты:" << endl;
    cout << "Мин: " << seqMin << ", Макс: " << seqMax << endl;
    cout << "Время: " << seqDuration.count() << " секунд" << endl;

    // --- Задание 3: Параллельное измерение ---
    cout << "\nЗапуск параллельного поиска (OpenMP)..." << endl;
    
    start = chrono::high_resolution_clock::now();
    
    parallelMinMax(arr, parMin, parMax);
    
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> parDuration = end - start;

    cout << "Параллельные результаты (OpenMP):" << endl;
    cout << "Мин: " << parMin << ", Макс: " << parMax << endl;
    cout << "Время: " << parDuration.count() << " секунд" << endl;
    
    // Вычисление ускорения (Speedup = T_seq / T_par)
    cout << "Ускорение: " << seqDuration.count() / parDuration.count() << "x" << endl;

    return 0;
}
