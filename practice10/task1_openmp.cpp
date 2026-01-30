#include <iostream>
#include <vector>
#include <omp.h>
#include <numeric>
#include <cmath>
#include <iomanip>

// Функция для вычисления суммы массива (Параллельная версия)
// Function to calculate array sum (Parallel version)
double calculate_sum(const std::vector<double>& data) {
    double sum = 0.0;
    // Используем OpenMP reduction для суммирования
    // Using OpenMP reduction for summation
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < data.size(); ++i) {
        sum += data[i]; // Добавляем элемент к общей сумме
    }
    return sum;
}

// Функция для вычисления дисперсии (Параллельная версия)
// Function to calculate variance (Parallel version)
double calculate_variance(const std::vector<double>& data, double mean) {
    double sum_sq_diff = 0.0;
    // Вычисляем сумму квадратов разностей параллельно
    // Calculate sum of squared differences in parallel
    #pragma omp parallel for reduction(+:sum_sq_diff)
    for (int i = 0; i < data.size(); ++i) {
        double diff = data[i] - mean;
        sum_sq_diff += diff * diff; // Накапливаем квадрат разности
    }
    return sum_sq_diff / data.size();
}

int main() {
    // Размер массива данных (100 миллионов элементов)
    // Data array size (100 million elements)
    const size_t N = 100000000; 
    std::cout << "Initializing array with " << N << " elements..." << std::endl;

    // Выделение памяти и инициализация массива
    // Memory allocation and array initialization
    std::vector<double> data(N);
    
    // Параллельная инициализация
    // Parallel initialization
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        data[i] = sin(i) * cos(i); // Заполняем некоторыми значениями
    }

    std::cout << "Array initialized." << std::endl;

    // Тестирование с различным количеством потоков
    // Testing with different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    // Переменная для хранения времени выполнения последовательной версии (1 поток)
    // Variable to store serial execution time (1 thread)
    double serial_time = 0.0;

    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << "| Threads | Time (s) | Speedup | Parallel Part (Est) |" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int threads : thread_counts) {
        // Установка количества потоков
        // Set number of threads
        omp_set_num_threads(threads);

        // Начало замера времени
        // Start timing
        double start_time = omp_get_wtime();

        // Вычисления
        // Calculations
        double sum = calculate_sum(data);
        double mean = sum / N;
        double variance = calculate_variance(data, mean);

        // Конец замера времени
        // End timing
        double end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;

        // Если это первый запуск (1 поток), сохраняем время как базовое
        // If this is the first run (1 thread), save time as baseline
        if (threads == 1) {
            serial_time = elapsed_time;
        }

        double speedup = serial_time / elapsed_time;
        
        // Оценка параллельной части по закону Амдала (приближенно)
        // Estimating parallel portion using Amdahl's Law (approximate)
        // Speedup = 1 / ((1 - P) + P/N) => P = ... 
        // Это упрощенная оценка
        double parallel_fraction = 0.0;
        if (threads > 1 && speedup > 1.0) {
             parallel_fraction = (1.0 / speedup - 1.0) / (1.0 / threads - 1.0);
        }

        std::cout << "| " << std::setw(7) << threads << " | " 
                  << std::setw(8) << std::fixed << std::setprecision(4) << elapsed_time << " | "
                  << std::setw(7) << std::fixed << std::setprecision(2) << speedup << " | "
                  << std::setw(19) << std::fixed << std::setprecision(2) << (threads == 1 ? 1.00 : parallel_fraction) << " |" 
                  << std::endl;
        
        // Вывод результатов для проверки (только для первого запуска, чтобы не засорять)
        // Output results for verification (only for first run to avoid clutter)
        if (threads == 1) {
             std::cout << "  (Debug: Sum=" << sum << ", Mean=" << mean << ", Var=" << variance << ")" << std::endl;
        }
    }
    std::cout << "------------------------------------------------------------" << std::endl;

    return 0;
}
