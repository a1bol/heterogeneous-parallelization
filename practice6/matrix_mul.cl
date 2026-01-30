// OpenCL Kernel for Matrix Multiplication
// Ядро OpenCL для умножения матриц

__kernel void matrix_mul(__global const float* A, 
                       __global const float* B, 
                       __global float* C, 
                       int N, int M, int K) {
    
    // Получение глобальных идентификаторов (строка и столбец)
    // Get global IDs (row and col)
    int row = get_global_id(1); // Y-axis corresponds to row
    int col = get_global_id(0); // X-axis corresponds to col

    // Проверка границ
    // Boundary check
    if (row < N && col < K) {
        float sum = 0.0f;
        
        // Вычисление скалярного произведения строки A и столбца B
        // Calculate dot product of row A and col B
        for (int k = 0; k < M; k++) {
            sum += A[row * M + k] * B[k * K + col];
        }
        
        // Запись результата
        // Write result
        C[row * K + col] = sum;
    }
}
