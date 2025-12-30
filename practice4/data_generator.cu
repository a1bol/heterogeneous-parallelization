#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;

// Программа для генерации массива случайных чисел
// Используется для подготовки тестовых данных

int main() {
    srand(time(0));
    
    // Размер массива: 1,000,000 элементов
    const int SIZE = 1000000;
    
    cout << "=== Data Generator ===" << endl;
    cout << "Generating array of " << SIZE << " random integers..." << endl;
    
    // Выделяем память для массива
    int* data = new int[SIZE];
    
    // Заполняем массив случайными числами от 0 до 999
    for (int i = 0; i < SIZE; ++i) {
        data[i] = rand() % 1000;
    }
    
    cout << "✓ Array generated successfully" << endl;
    
    // Выводим первые 10 элементов для проверки
    cout << "\nFirst 10 elements: ";
    for (int i = 0; i < 10; ++i) {
        cout << data[i] << " ";
    }
    cout << endl;
    
    // Выводим последние 10 элементов
    cout << "Last 10 elements: ";
    for (int i = SIZE - 10; i < SIZE; ++i) {
        cout << data[i] << " ";
    }
    cout << endl;
    
    // Вычисляем базовую статистику
    long long sum = 0;
    int min_val = data[0];
    int max_val = data[0];
    
    for (int i = 0; i < SIZE; ++i) {
        sum += data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    double average = (double)sum / SIZE;
    
    cout << "\n=== Statistics ===" << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << average << endl;
    cout << "Min: " << min_val << endl;
    cout << "Max: " << max_val << endl;
    
    // Опционально: сохранение в файл
    cout << "\nSave to file? (y/n): ";
    char choice;
    cin >> choice;
    
    if (choice == 'y' || choice == 'Y') {
        ofstream outfile("data.txt");
        if (outfile.is_open()) {
            outfile << SIZE << endl; // Первая строка - размер
            for (int i = 0; i < SIZE; ++i) {
                outfile << data[i] << " ";
                if ((i + 1) % 100 == 0) outfile << endl; // Для читаемости
            }
            outfile.close();
            cout << "✓ Data saved to data.txt" << endl;
        } else {
            cout << "✗ Error: Could not open file for writing" << endl;
        }
    }
    
    // Освобождаем память
    delete[] data;
    
    cout << "\n✓ Program completed successfully" << endl;
    
    return 0;
}
