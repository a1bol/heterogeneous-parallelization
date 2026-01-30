#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Функция для чтения текста ядра из файла
// Function to read kernel source from file
string readKernelFile(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open kernel file: " << filename << endl;
        exit(1);
    }
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

// Проверка ошибок OpenCL
// OpenCL error checking
void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        cerr << "Error during " << operation << ": " << err << endl;
        exit(1);
    }
}

// Основная функция для запуска теста на конкретном устройстве
// Main function to run calculation on a specific device
void run_test(cl_platform_id platform, cl_device_id device, const char* device_name, int n) {
    cl_int err;

    cout << "\nRunning on Device: " << device_name << endl;

    // 1. Создание контекста
    // Create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");

    // 2. Создание очереди команд
    // Create command queue
    // Note: In OpenCL 2.0+ clCreateCommandQueueWithProperties is used, but clCreateCommandQueue is compatible
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    // 3. Компиляция программы
    // Build program
    string kernelSource = readKernelFile("vector_add.cl");
    const char* sourceCStr = kernelSource.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &sourceCStr, NULL, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Если ошибка компиляции, выводим лог
        // If build error, show log
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), NULL);
        cerr << "Build Log:\n" << buildLog.data() << endl;
        exit(1);
    }

    // 4. Создание ядра
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    checkError(err, "clCreateKernel");

    // 5. Подготовка данных
    // Prepare data
    size_t bytes = n * sizeof(float);
    vector<float> h_A(n);
    vector<float> h_B(n);
    vector<float> h_C(n);

    // Инициализация массивов
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 6. Выделение памяти на устройстве
    // Allocate device memory
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A.data(), &err);
    checkError(err, "clCreateBuffer A");
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B.data(), &err);
    checkError(err, "clCreateBuffer B");
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    checkError(err, "clCreateBuffer C");

    // 7. Установка аргументов ядра
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    checkError(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    checkError(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    checkError(err, "clSetKernelArg 2");

    // 8. Запуск ядра и замер времени
    // Execute kernel and measure time
    size_t globalSize = n;
    size_t localSize = 256; // Размер рабочей группы / Work group size
    // Округляем globalSize до кратного localSize, если нужно, или даем OpenCL решить (NULL)
    // Round globalSize or let OpenCL decide. For simplicity, we ensure n is multiple or use NULL for local if supported well.
    // Для CPU localSize может быть другим. OpenCL позволяет передать NULL для localSize.
    // For CPU localSize might differ. OpenCL allows NULL. 

    // Используем таймер C++
    // Using C++ timer
    auto start = high_resolution_clock::now();

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    checkError(err, "clEnqueueNDRangeKernel");

    // Ждем завершения всех команд
    // Wait for finish
    clFinish(queue);

    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;

    cout << "Execution time: " << time_ms << " ms" << endl;

    // 9. Чтение результата
    // Read result
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, h_C.data(), 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer");

    // 10. Проверка (частичная)
    // Verification (partial)
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            cout << "Mismatch at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << endl;
            break;
        }
    }
    if (correct) {
        cout << "Result: CORRECT" << endl;
    } else {
        cout << "Result: INCORRECT" << endl;
    }

    // Очистка ресурсов
    // Cleanup
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    // Количество элементов
    // Number of elements
    const int N = 10000000; 
    cout << "Vector Addition Test (N = " << N << ")" << endl;

    // Получение платформ
    // Get platforms
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        cerr << "No OpenCL platforms found." << endl;
        return 1;
    }

    vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), NULL);

    for (const auto& platform : platforms) {
        char platformName[128];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
        cout << "Platform: " << platformName << endl;

        // Получение устройств для платформы
        // Get devices for platform
        cl_uint numDevices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) continue;

        vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);

        for (const auto& device : devices) {
            char deviceName[128];
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            cl_device_type deviceType;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);

            string typeStr = (deviceType & CL_DEVICE_TYPE_CPU) ? "CPU" : 
                             (deviceType & CL_DEVICE_TYPE_GPU) ? "GPU" : "Other";
            
            cout << "  Found Device: " << deviceName << " (" << typeStr << ")" << endl;
            
            // Запуск теста на устройстве
            // Run test on device
            run_test(platform, device, deviceName, N);
        }
    }

    return 0;
}
