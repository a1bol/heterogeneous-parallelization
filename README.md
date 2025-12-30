# Heterogeneous Parallelization

Этот репозиторий содержит практические работы и задания по курсу "Гетерогенные вычисления".

## Навигация

### Задания (Assignments)

*   **[Задание 1 (ass1)](ass1/README.md)**
    *   **Описание**: Основы C++ и OpenMP. Динамическая память, параллельный поиск Мин/Макс, вычисление среднего значения.
    *   **Файлы**: `task1.cpp`, `task2_3.cpp`, `task4.cpp`

*   **[Задание 2 (ass2)](ass2/README.md)**
    *   **Описание**: Гетерогенная параллелизация с OpenMP и CUDA. Поиск мин/макс (OpenMP), Сортировка выбором (OpenMP), Сортировка слиянием (CUDA).
    *   **Файлы**: `task2.cpp`, `task3.cpp`, `task4.cu`

### Практические работы (Practices)

*   **[Практика 1 (practice1)](practice1/README.md)**
    *   **Описание**: Динамическая память и OpenMP. Параллельное вычисление суммы элементов массива.
    *   **Файлы**: `lab3.cpp`

*   **[Практика 2 (practice2)](practice2/README.md)**
    *   **Описание**: Параллельные алгоритмы сортировки с OpenMP. Сравнение производительности: Bubble Sort vs Odd-Even Sort, Selection Sort (параллельный поиск минимума).
    *   **Файлы**: `sorting.cpp`

*   **[Практика 3 (practice3)](practice3/README.md)**
    *   **Описание**: Параллельные алгоритмы сортировки на CUDA. Реализация и сравнение Merge Sort, Quick Sort (Hybrid), Heap Sort.
    *   **Файлы**: `merge_sort.cu`, `quick_sort.cu`, `heap_sort.cu`

*   **[Практика 4 (practice4)](practice4/README.md)**
    *   **Описание**: Оптимизация памяти GPU. Использование Shared Memory для ускорения редукции и сортировки.
    *   **Файлы**: `reduction.cu`, `bubble_sort.cu`, `data_generator.cu`
