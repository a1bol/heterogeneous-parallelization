/*
 * ============================================================================
 * СРАВНЕНИЕ СТРАТЕГИЙ ПЛАНИРОВАНИЯ OpenMP
 * ============================================================================
 *
 * Программа демонстрирует разницу между тремя стратегиями планирования:
 * - schedule(static)  - статическое распределение итераций
 * - schedule(dynamic) - динамическое распределение итераций
 * - schedule(guided)  - управляемое распределение итераций
 *
 * Для демонстрации используется алгоритм с НЕРАВНОМЕРНОЙ нагрузкой:
 * проверка чисел на простоту, где большие числа требуют больше вычислений.
 *
 * ============================================================================
 */

#include <iostream> // Стандартный ввод-вывод (cout, cin)
#include <vector>   // Контейнер vector (динамический массив)
#include <cmath>    // Математические функции (sqrt)
#include <chrono>   // Высокоточное измерение времени
#include <iomanip>  // Форматирование вывода (setw, setprecision)
#include <omp.h>    // Библиотека OpenMP для параллельного программирования
#ifdef _WIN32
#include <windows.h> // Для SetConsoleOutputCP (русский вывод в Windows)
#endif

using namespace std;

// ============================================================================
// ГЛОБАЛЬНЫЕ КОНСТАНТЫ
// ============================================================================
const int NUM_THREADS = 4;       // Фиксированное число потоков для тестирования
const int ARRAY_SIZE = 50000;    // Размер массива чисел для проверки
const int MAX_NUMBER = 10000000; // Максимальное число (10 миллионов)
const int CHUNK_SIZE = 100;      // Размер блока для dynamic/guided

// ============================================================================
// ФУНКЦИЯ ПРОВЕРКИ ЧИСЛА НА ПРОСТОТУ
// ============================================================================
// Сложность: O(sqrt(n)) - для числа n
// Создает НЕРАВНОМЕРНУЮ нагрузку: большие числа требуют больше итераций
// Например: для n=100 нужно ~5 проверок, для n=10000000 нужно ~1500 проверок
bool isPrime(int n)
{
    // Числа меньше 2 не являются простыми
    if (n < 2)
        return false;
    // 2 - единственное четное простое число
    if (n == 2)
        return true;
    // Все четные числа > 2 не простые
    if (n % 2 == 0)
        return false;

    // Проверяем делители от 3 до sqrt(n) с шагом 2
    int sqrtN = static_cast<int>(sqrt(static_cast<double>(n)));
    for (int i = 3; i <= sqrtN; i += 2)
    {
        if (n % i == 0)
            return false;
    }
    return true;
}

// ============================================================================
// ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ (базовая для сравнения)
// ============================================================================
int countPrimesSequential(const vector<int> &numbers)
{
    int count = 0;
    // Простой последовательный цикл - один поток обрабатывает все элементы
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        if (isPrime(numbers[i]))
            count++;
    }
    return count;
}

// ============================================================================
// ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ: schedule(static)
// ============================================================================
// СТАТИЧЕСКОЕ ПЛАНИРОВАНИЕ:
// - Итерации делятся ПОРОВНУ между потоками ДО начала выполнения
// - Пример: 100 итераций, 4 потока -> поток 0: 0-24, поток 1: 25-49, и т.д.
//
// ПРОБЛЕМА для нашей задачи:
// - Числа в конце массива БОЛЬШЕ и требуют БОЛЬШЕ вычислений
// - Последний поток получит самую тяжелую работу
// - Остальные потоки будут ПРОСТАИВАТЬ, ожидая последний
int countPrimesStatic(const vector<int> &numbers)
{
    int count = 0;
// schedule(static) - статическое распределение
// reduction(+:count) - безопасное суммирование из всех потоков
#pragma omp parallel for schedule(static) reduction(+ : count) num_threads(NUM_THREADS)
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        if (isPrime(numbers[i]))
            count++;
    }
    return count;
}

// ============================================================================
// ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ: schedule(static, chunk_size)
// ============================================================================
// СТАТИЧЕСКОЕ С РАЗМЕРОМ БЛОКА:
// - Итерации делятся на блоки размера chunk_size
// - Блоки распределяются ЦИКЛИЧЕСКИ (round-robin)
// - Пример: chunk=100, 4 потока:
//   Поток 0: итерации 0-99, 400-499, 800-899...
//   Поток 1: итерации 100-199, 500-599, 900-999...
//
// ПРЕИМУЩЕСТВО: лучше балансирует нагрузку при неравномерных итерациях
int countPrimesStaticChunk(const vector<int> &numbers)
{
    int count = 0;
// schedule(static, CHUNK_SIZE) - статическое с блоками
#pragma omp parallel for schedule(static, CHUNK_SIZE) reduction(+ : count) num_threads(NUM_THREADS)
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        if (isPrime(numbers[i]))
            count++;
    }
    return count;
}

// ============================================================================
// ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ: schedule(dynamic)
// ============================================================================
// ДИНАМИЧЕСКОЕ ПЛАНИРОВАНИЕ:
// - Итерации распределяются ВО ВРЕМЯ выполнения
// - Когда поток заканчивает свой блок, он ЗАПРАШИВАЕТ следующий
// - Нет заранее определенного распределения
//
// КОГДА ИСПОЛЬЗОВАТЬ:
// + Когда итерации имеют СИЛЬНО различающуюся нагрузку
// + Когда нельзя предсказать время выполнения итерации
//
// НЕДОСТАТОК: накладные расходы на синхронизацию при каждом запросе
int countPrimesDynamic(const vector<int> &numbers)
{
    int count = 0;
// schedule(dynamic, CHUNK_SIZE) - динамическое распределение
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE) reduction(+ : count) num_threads(NUM_THREADS)
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        if (isPrime(numbers[i]))
            count++;
    }
    return count;
}

// ============================================================================
// ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ: schedule(guided)
// ============================================================================
// УПРАВЛЯЕМОЕ ПЛАНИРОВАНИЕ:
// - КОМБИНАЦИЯ static и dynamic
// - Размер блока УМЕНЬШАЕТСЯ со временем
// - В начале: большие блоки (меньше накладных расходов)
// - В конце: маленькие блоки (лучше балансировка)
// - Формула: размер = оставшиеся_итерации / число_потоков
//
// ПРЕИМУЩЕСТВО: хороший баланс между накладными расходами и балансировкой
int countPrimesGuided(const vector<int> &numbers)
{
    int count = 0;
// schedule(guided, CHUNK_SIZE) - CHUNK_SIZE это минимальный размер блока
#pragma omp parallel for schedule(guided, CHUNK_SIZE) reduction(+ : count) num_threads(NUM_THREADS)
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        if (isPrime(numbers[i]))
            count++;
    }
    return count;
}

// ============================================================================
// ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ
// ============================================================================
// Создаем массив с ВОЗРАСТАЮЩИМИ числами для демонстрации неравномерности:
// - Первые итерации: малые числа -> быстрая проверка
// - Последние итерации: большие числа -> медленная проверка
vector<int> generateTestData(int size, int maxNumber)
{
    vector<int> numbers(size);
    for (int i = 0; i < size; ++i)
    {
        // Линейное распределение от 2 до maxNumber
        // Используем long long чтобы избежать переполнения int
        numbers[i] = 2 + static_cast<int>((static_cast<long long>(i) * (maxNumber - 2)) / size);
    }
    return numbers;
}

// ============================================================================
// ИЗМЕРЕНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ
// ============================================================================
template <typename Func>
double measureTime(Func function, const vector<int> &numbers, int &result)
{
    auto start = chrono::high_resolution_clock::now(); // Начало
    result = function(numbers);                        // Выполнение
    auto end = chrono::high_resolution_clock::now();   // Конец
    chrono::duration<double, milli> duration = end - start;
    return duration.count(); // Возвращаем время в миллисекундах
}

// ============================================================================
// ГЛАВНАЯ ФУНКЦИЯ
// ============================================================================
int main()
{
    // Установка кодировки UTF-8 для корректного вывода русского текста в Windows
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    cout << "============================================================" << endl;
    cout << "   СРАВНЕНИЕ СТРАТЕГИЙ ПЛАНИРОВАНИЯ OpenMP" << endl;
    cout << "============================================================" << endl;
    cout << endl;

    // Вывод конфигурации теста
    cout << "Конфигурация теста:" << endl;
    cout << "  - Количество потоков: " << NUM_THREADS << endl;
    cout << "  - Размер массива: " << ARRAY_SIZE << " элементов" << endl;
    cout << "  - Диапазон чисел: 2 - " << MAX_NUMBER << endl;
    cout << "  - Размер блока (chunk): " << CHUNK_SIZE << endl;
    cout << endl;

    // Информация о системе
    cout << "Информация о системе:" << endl;
    cout << "  - Макс. потоков OpenMP: " << omp_get_max_threads() << endl;
    cout << "  - Количество процессоров: " << omp_get_num_procs() << endl;
    cout << endl;

    // Генерация тестовых данных
    cout << "Генерация тестовых данных..." << endl;
    vector<int> numbers = generateTestData(ARRAY_SIZE, MAX_NUMBER);
    cout << "  Первые 5 чисел: ";
    for (int i = 0; i < 5; ++i)
        cout << numbers[i] << " ";
    cout << endl;
    cout << "  Последние 5 чисел: ";
    for (int i = ARRAY_SIZE - 5; i < ARRAY_SIZE; ++i)
        cout << numbers[i] << " ";
    cout << endl
         << endl;

    // Переменные для результатов
    int resultSeq, resultStatic, resultStaticChunk, resultDynamic, resultGuided;
    double timeSeq, timeStatic, timeStaticChunk, timeDynamic, timeGuided;

    // ========================================================================
    // ЗАПУСК ТЕСТОВ
    // ========================================================================
    cout << "============================================================" << endl;
    cout << "                   РЕЗУЛЬТАТЫ ТЕСТОВ" << endl;
    cout << "============================================================" << endl;
    cout << endl;

    // 1. Последовательная версия
    cout << "[1/5] Последовательная версия..." << endl;
    timeSeq = measureTime(countPrimesSequential, numbers, resultSeq);
    cout << "      Найдено простых чисел: " << resultSeq << endl;
    cout << "      Время: " << fixed << setprecision(2) << timeSeq << " мс" << endl;
    cout << endl;

    // 2. schedule(static)
    cout << "[2/5] schedule(static)..." << endl;
    timeStatic = measureTime(countPrimesStatic, numbers, resultStatic);
    cout << "      Найдено простых чисел: " << resultStatic << endl;
    cout << "      Время: " << fixed << setprecision(2) << timeStatic << " мс" << endl;
    cout << endl;

    // 3. schedule(static, chunk)
    cout << "[3/5] schedule(static, " << CHUNK_SIZE << ")..." << endl;
    timeStaticChunk = measureTime(countPrimesStaticChunk, numbers, resultStaticChunk);
    cout << "      Найдено простых чисел: " << resultStaticChunk << endl;
    cout << "      Время: " << fixed << setprecision(2) << timeStaticChunk << " мс" << endl;
    cout << endl;

    // 4. schedule(dynamic)
    cout << "[4/5] schedule(dynamic, " << CHUNK_SIZE << ")..." << endl;
    timeDynamic = measureTime(countPrimesDynamic, numbers, resultDynamic);
    cout << "      Найдено простых чисел: " << resultDynamic << endl;
    cout << "      Время: " << fixed << setprecision(2) << timeDynamic << " мс" << endl;
    cout << endl;

    // 5. schedule(guided)
    cout << "[5/5] schedule(guided, " << CHUNK_SIZE << ")..." << endl;
    timeGuided = measureTime(countPrimesGuided, numbers, resultGuided);
    cout << "      Найдено простых чисел: " << resultGuided << endl;
    cout << "      Время: " << fixed << setprecision(2) << timeGuided << " мс" << endl;
    cout << endl;

    // ========================================================================
    // СРАВНИТЕЛЬНЫЙ АНАЛИЗ
    // ========================================================================
    cout << "============================================================" << endl;
    cout << "              СРАВНИТЕЛЬНЫЙ АНАЛИЗ" << endl;
    cout << "============================================================" << endl;
    cout << endl;

    // Проверка корректности
    cout << "Проверка корректности:" << endl;
    if (resultSeq == resultStatic && resultStatic == resultStaticChunk &&
        resultStaticChunk == resultDynamic && resultDynamic == resultGuided)
    {
        cout << "  [OK] Все методы нашли одинаковое количество простых чисел: " << resultSeq << endl;
    }
    else
    {
        cout << "  [ERROR] Результаты не совпадают!" << endl;
    }
    cout << endl;

    // Таблица результатов
    cout << "Таблица времени выполнения:" << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << left << setw(25) << "Метод" << setw(15) << "Время (мс)" << setw(15) << "Ускорение" << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << left << setw(25) << "Sequential" << setw(15) << fixed << setprecision(2) << timeSeq << setw(15) << "1.00x (база)" << endl;
    cout << left << setw(25) << "static" << setw(15) << fixed << setprecision(2) << timeStatic << setw(15) << fixed << setprecision(2) << (timeSeq / timeStatic) << "x" << endl;
    cout << left << setw(25) << "static, chunk" << setw(15) << fixed << setprecision(2) << timeStaticChunk << setw(15) << fixed << setprecision(2) << (timeSeq / timeStaticChunk) << "x" << endl;
    cout << left << setw(25) << "dynamic" << setw(15) << fixed << setprecision(2) << timeDynamic << setw(15) << fixed << setprecision(2) << (timeSeq / timeDynamic) << "x" << endl;
    cout << left << setw(25) << "guided" << setw(15) << fixed << setprecision(2) << timeGuided << setw(15) << fixed << setprecision(2) << (timeSeq / timeGuided) << "x" << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << endl;

    // Определение лучшего метода
    double bestTime = timeStatic;
    string bestMethod = "schedule(static)";
    if (timeStaticChunk < bestTime)
    {
        bestTime = timeStaticChunk;
        bestMethod = "schedule(static, chunk)";
    }
    if (timeDynamic < bestTime)
    {
        bestTime = timeDynamic;
        bestMethod = "schedule(dynamic)";
    }
    if (timeGuided < bestTime)
    {
        bestTime = timeGuided;
        bestMethod = "schedule(guided)";
    }

    cout << "Лучший параллельный метод: " << bestMethod << " (" << fixed << setprecision(2) << bestTime << " мс)" << endl;
    cout << endl;

    // ========================================================================
    // ВЫВОДЫ
    // ========================================================================
    cout << "============================================================" << endl;
    cout << "                       ВЫВОДЫ" << endl;
    cout << "============================================================" << endl;
    cout << endl;

    cout << "1. schedule(static):" << endl;
    cout << "   - Делит итерации поровну между потоками заранее" << endl;
    cout << "   - При неравномерной нагрузке возникает дисбаланс" << endl;
    cout << "   - Последние потоки получают большие числа" << endl;
    cout << endl;

    cout << "2. schedule(static, chunk):" << endl;
    cout << "   - Итерации распределяются блоками циклически" << endl;
    cout << "   - Лучше балансирует нагрузку чем простой static" << endl;
    cout << "   - Низкие накладные расходы" << endl;
    cout << endl;

    cout << "3. schedule(dynamic):" << endl;
    cout << "   - Итерации распределяются во время выполнения" << endl;
    cout << "   - Отлично балансирует неравномерную нагрузку" << endl;
    cout << "   - Накладные расходы на синхронизацию" << endl;
    cout << endl;

    cout << "4. schedule(guided):" << endl;
    cout << "   - Комбинация static и dynamic" << endl;
    cout << "   - Размер блока уменьшается со временем" << endl;
    cout << "   - Хороший баланс между расходами и балансировкой" << endl;
    cout << endl;

    cout << "РЕКОМЕНДАЦИИ:" << endl;
    cout << "- Равномерная нагрузка -> schedule(static)" << endl;
    cout << "- Неравномерная нагрузка -> schedule(dynamic) или schedule(guided)" << endl;
    cout << "- Если не уверены -> schedule(guided) как универсальный выбор" << endl;
    cout << endl;

    cout << "============================================================" << endl;
    cout << "               ПРОГРАММА ЗАВЕРШЕНА" << endl;
    cout << "============================================================" << endl;

    return 0;
}
