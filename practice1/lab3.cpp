#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Function to find the mean of array elements (sequential version for logic check usually, but not strictly required by prompt parallel part)
// Requirement 2: Write a function for finding the mean of array elements.
double calculateMean(int* arr, int size) {
    long long sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return static_cast<double>(sum) / size;
}

int main() {
    // Seed for random number generation
    srand(static_cast<unsigned>(time(0)));

    int size;
    cout << "Enter the size of the array: ";
    if (!(cin >> size) || size <= 0) {
        cerr << "Invalid size. Please enter a positive integer." << endl;
        return 1;
    }

    // 1. Create a dynamic array using pointers
    int* arr = new int[size];

    // 2. Fill the array with random numbers
    cout << "Filling array with random numbers..." << endl;
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 100; // Random numbers between 0 and 99
    }

    // 3. Parallel calculation of average
    // a. Use #pragma omp parallel for reduction(+:sum) for parallel summation
    long long sum = 0;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }

    double mean = static_cast<double>(sum) / size;

    cout << "Total Sum (Parallel): " << sum << endl;
    cout << "Average (Parallel): " << mean << endl;

    // Optional: Call the sequential function to demonstrate it exists as per point 2
    // cout << "Average (Sequential Function): " << calculateMean(arr, size) << endl;

    // 4. Free memory
    delete[] arr;
    cout << "Memory successfully freed." << endl;

    return 0;
}
