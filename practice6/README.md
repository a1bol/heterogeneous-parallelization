# Практическая работа 6: Программирование на OpenCL

## Описание
Разработка кросс-платформенного приложения для параллельных вычислений на CPU и GPU с использованием OpenCL.
Задачи:
1. Векторное сложение.
2. Умножение матриц.

## Инструкция по запуску
Для компиляции требуется установленный CUDA Toolkit (или OpenCL SDK) и компилятор C++ (MSVC/GCC/Clang).

### Компиляция и Запуск (Windows)
Самый простой способ - компилировать и запускать прямо в папке с кодом.

### Компиляция и Запуск (Windows)
**PowerShell:**
```powershell
g++ -o vector_add vector_add.cpp -I "$env:CUDA_PATH\include" -L "$env:CUDA_PATH\lib\x64" -lOpenCL -DCL_TARGET_OPENCL_VERSION=300
g++ -o matrix_mul matrix_mul.cpp -I "$env:CUDA_PATH\include" -L "$env:CUDA_PATH\lib\x64" -lOpenCL -DCL_TARGET_OPENCL_VERSION=300
```

**CMD (Командная строка):**
```cmd
g++ -o vector_add vector_add.cpp -I "%CUDA_PATH%\include" -L "%CUDA_PATH%\lib\x64" -lOpenCL -DCL_TARGET_OPENCL_VERSION=300
g++ -o matrix_mul matrix_mul.cpp -I "%CUDA_PATH%\include" -L "%CUDA_PATH%\lib\x64" -lOpenCL -DCL_TARGET_OPENCL_VERSION=300
```

3. Запустите:
```powershell
./vector_add
./matrix_mul
```
```cmd
./vector_add
./matrix_mul
```

*Важно: Файлы `.cl` (ядра) должны находиться в той же папке, откуда вы запускаете программу.*

---

## Отчет о результатах
(Примерные данные, заполните после запуска)

| Device | Task | Time (ms) | Status |
|--------|------|-----------|--------|
| AMD gfx90c (GPU) | Vector Add | 13.025 | Correct |
| NVIDIA GTX 1650 (GPU)| Vector Add | 28.392 | Correct |
| AMD gfx90c (GPU) | Matrix Mul | 4 | Consistent |
| NVIDIA GTX 1650 (GPU)| Matrix Mul | 3 | Consistent |

---

## Контрольные вопросы

1. **Какие основные типы памяти используются в OpenCL?**
   - **Global Memory**: Доступна всем рабочим элементам (медленная, большая).
   - **Constant Memory**: Только для чтения, кэшируется.
   - **Local Memory**: Общая для рабочей группы (быстрая).
   - **Private Memory**: Память одного рабочего элемента (регистры).

2. **Как настроить глобальную и локальную рабочую группу?**
   - Глобальный размер (`global_work_size`) определяет общее количество потоков.
   - Локальный размер (`local_work_size`) определяет количество потоков в одной группе, которые могут использовать общую локальную память и синхронизироваться.
   - Задается в функции `clEnqueueNDRangeKernel`.

3. **Чем отличается OpenCL от CUDA?**
   - **OpenCL (Open Computing Language)**: Открытый стандарт, работает на CPU, GPU, DSP, FPGA разных производителей (Intel, AMD, NVIDIA, ARM).
   - **CUDA (Compute Unified Device Architecture)**: Проприетарная технология NVIDIA, работает только на GPU NVIDIA, но обычно более проста в настройке и имеет развитую экосистему библиотек.

4. **Какие преимущества дает использование OpenCL?**
   - **Переносимость (Portability)**: Код можно запускать на широком спектре устройств без переписывания.
   - **Гетерогенность**: Возможность использовать CPU и GPU одновременно для выполнения задачи.
