#include <iostream> // Ввод-вывод
#include <vector>   // Вектор (массив)
#include <cstdlib>  // Стандартная библиотека
#include <ctime>    // Время
#include <chrono>   // Измерение времени (chrono)
#include <omp.h>    // OpenMP

using namespace std; // Стандартное пр-во имен

// Задание 4: Вычисление среднего значения массива (5 млн элементов)
// Демонстрация использования OpenMP Reduction для суммирования

// --- ПОСЛЕДОВАТЕЛЬНАЯ ФУНКЦИЯ ---
double sequentialAverage(const vector<int>& arr) {
    long long sum = 0; // long long для предотвращения переполнения
    // Простой цикл суммирования
    for (int val : arr) {
        sum += val;
    }
    // Возвращаем сумму, деленную на размер (среднее арифметическое)
    return static_cast<double>(sum) / arr.size();
}

// --- ПАРАЛЛЕЛЬНАЯ ФУНКЦИЯ (OpenMP Reduction) ---
double parallelAverage(const vector<int>& arr) {
    long long sum = 0;
    
    // Директива #pragma omp parallel for reduction(+:sum) делает две вещи:
    // 1. parallel for: Распределяет итерации цикла между потоками.
    // 2. reduction(+:sum): Создает локальную копию переменной 'sum' для каждого потока.
    //    Каждый поток суммирует свои элементы в свою локальную копию.
    //    В конце работы цикла все локальные копии суммируются в глобальную переменную 'sum'.
    //    Это самый эффективный и безопасный способ суммирования в OpenMP (без явных критических секций).
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    
    return static_cast<double>(sum) / arr.size();
}

int main() {
    // Инициализация генератора
    srand(static_cast<unsigned int>(time(0)));
    
    const int SIZE = 5000000; // 5 миллионов элементов
    cout << "Генерация массива из " << SIZE << " целых чисел..." << endl;
    
    // Выделение памяти под вектор
    vector<int> arr(SIZE);
    
    // Заполнение случайными значениями
    for (int& val : arr) {
        val = (rand() % 100) + 1; // Числа от 1 до 100
    }

    double seqAvg, parAvg;

    // --- Запуск последовательной версии ---
    cout << "\nЗапуск последовательного вычисления..." << endl;
    auto start = chrono::high_resolution_clock::now();
    
    seqAvg = sequentialAverage(arr);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> seqDuration = end - start;

    cout << "Последовательные результаты:" << endl;
    cout << "Среднее: " << seqAvg << endl;
    cout << "Время: " << seqDuration.count() << " секунд" << endl;

    // --- Запуск параллельной версии (Reduction) ---
    cout << "\nЗапуск параллельного вычисления (OpenMP Reduction)..." << endl;
    start = chrono::high_resolution_clock::now();
    
    parAvg = parallelAverage(arr);
    
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> parDuration = end - start;

    cout << "Параллельные результаты:" << endl;
    cout << "Среднее: " << parAvg << endl;
    cout << "Время: " << parDuration.count() << " секунд" << endl;
    
    // Вывод ускорения
    cout << "Ускорение: " << seqDuration.count() / parDuration.count() << "x" << endl;

    return 0;
}
