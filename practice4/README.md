# Practice 4: Оптимизация использования памяти GPU

Эта практическая работа демонстрирует важность выбора типа памяти в CUDA программировании и влияние использования разделяемой памяти на производительность.

## Структура проекта

- [data_generator.cu](file:///c:/Users/Aibol/Desktop/AITU/2nd%20course/2nd%20trimester/heterogeneous-parallelization/practice4/data_generator.cu): Генератор тестовых данных (1,000,000 элементов)
- [reduction.cu](file:///c:/Users/Aibol/Desktop/AITU/2nd%20course/2nd%20trimester/heterogeneous-parallelization/practice4/reduction.cu): Сравнение редукции с разными типами памяти
- [bubble_sort.cu](file:///c:/Users/Aibol/Desktop/AITU/2nd%20course/2nd%20trimester/heterogeneous-parallelization/practice4/bubble_sort.cu): Оптимизированная сортировка с shared memory
- `README.md`: Этот файл

## Требования

- **NVIDIA CUDA Toolkit** - [Скачать](https://developer.nvidia.com/cuda-downloads)
- **NVIDIA GPU** (Compute Capability ≥ 3.0)
- **Visual Studio Build Tools** (для Windows)
- **Компилятор C++** с поддержкой CUDA

## Компиляция

### Windows (Developer Command Prompt for VS):

```bash
# Генератор данных
nvcc data_generator.cu -o data_generator.exe

# Редукция
nvcc reduction.cu -o reduction.exe

# Сортировка
nvcc bubble_sort.cu -o bubble_sort.exe
```

### Linux:

```bash
nvcc data_generator.cu -o data_generator
nvcc reduction.cu -o reduction
nvcc bubble_sort.cu -o bubble_sort
```

## Запуск

```bash
# 1. Сгенерировать тестовые данные (опционально)
./data_generator

# 2. Запустить редукцию с разными типами памяти
./reduction

# 3. Запустить оптимизированную сортировку
./bubble_sort
```

---

## Иерархия памяти GPU

### От самой быстрой к самой медленной:

| Тип памяти | Скорость | Размер | Видимость | Время жизни |
|------------|----------|--------|-----------|-------------|
| **Регистры** | ~1 cycle | ~64KB | Поток | Поток |
| **Shared Memory** | ~5-10 cycles | 16-96 KB | Блок | Блок |
| **L1 Cache** | ~20-40 cycles | 16-128 KB | SM | Авто |
| **L2 Cache** | ~200 cycles | 256KB-6MB | Устройство | Авто |
| **Global Memory** | ~400-800 cycles | GB | Устройство | Программа |

### Ключевые различия:

**Регистры:**
- Самые быстрые
- Автоматически используются компилятором
- Ограниченное количество → может снизить occupancy

**Shared Memory (Разделяемая):**
- Программируемый кэш
- Доступна всем потокам в блоке
- Идеальна для данных, используемых многократно
- Требует явного управления

**Global Memory (Глобальная):**
- Большой объем
- Медленная (400-800 циклов латентности)
- Доступна всем потокам
- Нужно минимизировать обращения

---

## Задание 1: Генерация данных

### Программа: `data_generator.cu`

**Функциональность:**
- Генерирует массив из 1,000,000 случайных целых чисел
- Вычисляет базовую статистику (сумма, среднее, мин, макс)
- Опционально сохраняет данные в файл

**Использование:**
```bash
./data_generator
```

**Вывод:**
- Первые 10 элементов
- Последние 10 элементов
- Статистика массива
- Опция сохранения в `data.txt`

---

## Задание 2: Оптимизация редукции

### Программа: `reduction.cu`

### Вариант 1: Только глобальная память

```cuda
__global__ void reductionGlobalMemory(int* input, int* output, int n) {
    int local_sum = 0;
    for (int i = tid; i < n; i += stride) {
        local_sum += input[i];
    }
    // ПРОБЛЕМА: atomicAdd в глобальную память - медленно!
    atomicAdd(&output[0], local_sum);
}
```

**Проблемы:**
- Каждый поток пишет в глобальную память
- Много конфликтов при atomic операциях
- Высокая латентность доступа

### Вариант 2: Глобальная + Разделяемая память

```mermaid
flowchart TD
    start([Начало]) --> global[Глобальная память]
    global --> load[Загрузка в Shared Memory]
    load --> sync1[__syncthreads]
    sync1 --> reduction[Параллельная редукция в Shared Mem]
    reduction --> sync2[__syncthreads]
    sync2 --> result{tid == 0?}
    result -- Да --> write[AtomicAdd в Глобальную память]
    result -- Нет --> end([Конец потока])
    write --> end
```

```cuda
__global__ void reductionSharedMemory(int* input, int* output, int n) {
    extern __shared__ int shared_data[];
    
    // 1. Каждый поток суммирует свои элементы
    int local_sum = /* ... */;
    
    // 2. Записываем в БЫСТРУЮ shared memory
    shared_data[tid] = local_sum;
    __syncthreads();
    
    // 3. Параллельная редукция в shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // 4. Только ОДНА запись на блок в глобальную память!
    if (tid == 0) atomicAdd(&output[0], shared_data[0]);
}
```

**Преимущества:**
- Редукция происходит в быстрой shared memory
- Вместо N записей в глобальную память - только количество блоков
- Уменьшение конфликтов atomic операций
- Значительное ускорение

### Ожидаемые результаты:

| Размер массива | Global Only | Global + Shared | Ускорение |
|----------------|-------------|-----------------|-----------|
| 10,000 | ~50 μs | ~20 μs | ~2.5x |
| 100,000 | ~400 μs | ~120 μs | ~3.3x |
| 1,000,000 | ~3500 μs | ~900 μs | ~3.9x |

**Вывод:** Использование shared memory дает **2-4x ускорение**!

---

## Задание 3: Оптимизация сортировки

### Программа: `bubble_sort.cu`

### Стратегия оптимизации:

**1. Блочная сортировка в shared memory:**

```mermaid
flowchart TD
    start([Start Block]) --> load[Load Global -> Shared]
    load --> sort[Bubble Sort in Shared Memory]
    sort --> write[Write Shared -> Global]
    write --> end([End Block])
    
    style load fill:#f9f,stroke:#333
    style sort fill:#ccf,stroke:#333
    style write fill:#f9f,stroke:#333
```

```cuda
__global__ void bubbleSortBlocks(int* global_arr, int block_size) {
    extern __shared__ int shared_arr[];
    
    // Загружаем блок в shared memory
    shared_arr[tid] = global_arr[block_start + tid];
    __syncthreads();
    
    // Сортируем в БЫСТРОЙ shared memory
    if (tid == 0) {
        // Bubble sort на shared_arr
    }
    __syncthreads();
    
    // Записываем обратно в global
    global_arr[block_start + tid] = shared_arr[tid];
}
```

**Преимущества:**
- Все операции сравнения/обмена в быстрой памяти
- Минимум обращений к глобальной памяти
- Каждый блок работает независимо

**2. Слияние с использованием shared memory:**
```cuda
__global__ void mergeBlocks(int* global_arr, int width) {
    extern __shared__ int shared_mem[];
    
    // Загружаем два отсортированных подмассива в shared memory
    // Сливаем в shared memory
    // Записываем результат обратно
}
```

### Производительность:

| Размер | CPU Bubble | GPU Optimized | Ускорение |
|--------|------------|---------------|-----------|
| 10,000 | ~250 ms | ~15 ms | ~16x |
| 100,000 | Слишком медленно | ~120 ms | - |
| 1,000,000 | Не практично | ~1100 ms | - |

**Примечание:** Bubble sort не является оптимальным алгоритмом для GPU. Для производственного кода используйте merge sort, radix sort или bitonic sort.

---

## Задание 4: Измерение производительности

### Практические рекомендации:

**1. Использование CUDA Events для точного измерения:**
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// GPU kernel execution
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

**2. Построение графиков:**

Используйте данные из программ для построения графиков:
- **Ось X**: Размер массива (10K, 100K, 1M)
- **Ось Y**: Время выполнения (ms или μs)
- **Линии**: Разные варианты (Global, Shared, CPU)

**Инструменты для графиков:**
- Python (matplotlib)
- Excel / Google Sheets
- MATLAB
- Online plotting tools

### Пример данных для графика (Reduction):

```
Size     | Global (μs) | Shared (μs) | Speedup
---------|-------------|-------------|--------
10,000   | 50          | 20          | 2.5x
100,000  | 400         | 120         | 3.3x
1,000,000| 3,500       | 900         | 3.9x
```

---

## Ключевые выводы

### 1. Важность выбора памяти:
- **Shared memory до 100x быстрее** глобальной
- Критично для производительности GPU приложений
- Требует внимательного проектирования алгоритмов

### 2. Когда использовать shared memory:
✅ Данные используются несколько раз
✅ Необходима коммуникация между потоками блока
✅ Нужна редукция или другие коллективные операции
✅ Доступ к данным можно "переиспользовать"

❌ Не используйте shared memory если:
- Данные читаются один раз
- Нет взаимодействия между потоками
- Простая параллельная обработка элементов

### 3. Лучшие практики:

**Минимизируйте обращения к глобальной памяти:**
- Загружайте данные в shared memory
- Обрабатывайте в shared memory
- Записывайте результат обратно

**Используйте coalesced memory access:**
- Соседние потоки читают соседние адреса
- Значительно улучшает пропускную способность

**Избегайте bank conflicts в shared memory:**
- Shared memory разделена на 32 банка
- Одновременный доступ к одному банку из разных потоков = конфликт
- Используйте padding при необходимости

### 4. Оптимизация производительности:

```
Общая формула успеха GPU:
= Высокий параллелизм
+ Эффективное использование памяти
+ Минимизация divergence
+ Хорошая occupancy
```

---

## Дополнительные ресурсы

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [GPU Performance Analysis and Optimization](https://developer.nvidia.com/blog/cuda-pro-tip-how-optimize-bandwidth-limited-kernels/)
- [Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

---

## Профилирование

Для детального анализа производительности используйте:

**NVIDIA Nsight Compute:**
```bash
ncu --set full ./reduction
```

**NVIDIA Visual Profiler:**
- Визуальный анализ использования памяти
- Timeline выполнения ядер
- Bottleneck analysis

**Ключевые метрики:**
- Memory throughput (GB/s)
- Occupancy (%)
- Warp execution efficiency
- Shared memory bank conflicts

---

## Заключение

Эффективное использование иерархии памяти GPU - ключ к высокой производительности. Shared memory предоставляет программируемый кэш высокой скорости, который при правильном использовании может дать **2-10x ускорение** по сравнению с использованием только глобальной памяти.

**Главное правило:** Думайте о памяти при проектировании GPU алгоритмов!
