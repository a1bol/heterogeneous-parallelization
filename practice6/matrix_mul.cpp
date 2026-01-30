#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Размеры матриц
// Matrix dimensions
const int N_SIZE = 512; // Строки A / Rows A
const int M_SIZE = 512; // Столбцы A / Cols A (Rows B)
const int K_SIZE = 512; // Столбцы B / Cols B

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

// Запуск теста на устройстве
// Run test on device
void run_test(cl_platform_id platform, cl_device_id device, const char* device_name) {
    cl_int err;
    cout << "\nRunning on Device: " << device_name << endl;

    // 1. Создание контекста
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");

    // 2. Создание очереди команд
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    // 3. Компиляция программы
    string kernelSource = readKernelFile("matrix_mul.cl");
    const char* sourceCStr = kernelSource.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &sourceCStr, NULL, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), NULL);
        cerr << "Build Log:\n" << buildLog.data() << endl;
        exit(1);
    }

    // 4. Создание ядра
    cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);
    checkError(err, "clCreateKernel");

    // 5. Подготовка данных
    size_t size_A = N_SIZE * M_SIZE * sizeof(float);
    size_t size_B = M_SIZE * K_SIZE * sizeof(float);
    size_t size_C = N_SIZE * K_SIZE * sizeof(float);

    vector<float> h_A(N_SIZE * M_SIZE);
    vector<float> h_B(M_SIZE * K_SIZE);
    vector<float> h_C(N_SIZE * K_SIZE);
    vector<float> h_C_CPU(N_SIZE * K_SIZE); // Для проверки / For verification

    // Инициализация матриц случайными значениями
    // Initialize matrices
    for (int i = 0; i < N_SIZE * M_SIZE; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < M_SIZE * K_SIZE; i++) h_B[i] = rand() / (float)RAND_MAX;

    // 6. Выделение памяти на устройстве
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, h_A.data(), &err);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_B, h_B.data(), &err);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_C, NULL, &err);

    // 7. Установка аргументов
    int n = N_SIZE, m = M_SIZE, k = K_SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &n);
    clSetKernelArg(kernel, 4, sizeof(int), &m);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    // 8. Запуск ядра
    // Размерность: X = K_SIZE (столбцы C), Y = N_SIZE (строки C)
    size_t globalSize[2] = { (size_t)K_SIZE, (size_t)N_SIZE };
    
    auto start = high_resolution_clock::now();

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    checkError(err, "clEnqueueNDRangeKernel");

    clFinish(queue);

    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<milliseconds>(end - start).count();

    cout << "Execution time: " << time_ms << " ms" << endl;

    // 9. Чтение
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size_C, h_C.data(), 0, NULL, NULL);

    // 10. Проверка (только один элемент для скорости или простой цикл)
    // Verification (CPU implementation)
    cout << "Verifying on CPU (might take some time for large matrices)..." << endl;
    // Проверим несколько случайных точек
    // Check random points to save time
    bool correct = true;
    for(int i=0; i<10; ++i) {
        int r = rand() % N_SIZE;
        int c = rand() % K_SIZE;
        float sum = 0.0f;
        for(int j=0; j<M_SIZE; ++j) {
            sum += h_A[r * M_SIZE + j] * h_B[j * K_SIZE + c];
        }
        if (abs(h_C[r * K_SIZE + c] - sum) > 1e-3) {
            correct = false;
            cout << "Mismatch at [" << r << "," << c << "]: GPU=" << h_C[r * K_SIZE + c] << ", CPU=" << sum << endl;
            break;
        }
    }

    if (correct) cout << "Result: CONSISTENT (Random sampling check passed)" << endl;
    else cout << "Result: INCORRECT" << endl;

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
    srand(time(0));
    cout << "Matrix Multiplication Test (" << N_SIZE << "x" << M_SIZE << " * " << M_SIZE << "x" << K_SIZE << ")" << endl;

    // Платформы
    cl_uint numPlatforms;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    if(numPlatforms == 0) return 1;

    vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), NULL);

    for (const auto& platform : platforms) {
        char name[128];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, NULL);
        cout << "Platform: " << name << endl;

        cl_uint numDevices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if(numDevices == 0) continue;
        vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);

        for(const auto& dev : devices) {
             char devName[128];
             clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devName), devName, NULL);
             
             // Запуск
             run_test(platform, dev, devName);
        }
    }
    return 0;
}
