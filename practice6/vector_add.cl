// OpenCL Kernel for Vector Addition
// Ядро OpenCL для сложения двух векторов

__kernel void vector_add(__global const float* A, 
                       __global const float* B, 
                       __global float* C) {
    
    // Получение глобального идентификатора потока (индекса элемента)
    // Get global ID (index of the element)
    int id = get_global_id(0);

    // Выполнение операции сложения для данного индекса
    // Perform addition for the current index
    C[id] = A[id] + B[id];
}
