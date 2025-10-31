#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

// Error checking macros for CUDA
#define CUDA_CHECK_ERROR
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

// Safe CUDA call implementation
inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_CHECK_ERROR
    do
    {
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                    file, line, cudaGetErrorString(err));
            exit(-1);
        }
    } while (0);
#endif
    return;
}

// CUDA error checking implementation
inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_CHECK_ERROR
    do
    {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed at %s:%i : %s.\n",
                    file, line, cudaGetErrorString(err));
            exit(-1);
        }
        err = cudaDeviceSynchronize();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
                    file, line, cudaGetErrorString(err));
            exit(-1);
        }
    } while (0);
#endif
    return;
}

// Function to create random array
int *makeRandArray(const int size, const int seed)
{
    srand(seed);
    int *array = new int[size];
    for (int i = 0; i < size; i++)
    {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

// Kernel to compute histogram for radix sort - counts 0s and 1s for current bit
__global__ void computeHistogram(int *input, int *histogram, int bit, int size, int numBlocks)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory for block-level histogram (2 bins for 0 and 1)
    __shared__ int blockHistogram[2];

    // Initialize shared memory histogram
    if (threadIdx.x < 2)
    {
        blockHistogram[threadIdx.x] = 0;
    }
    __syncthreads();

    // Each thread processes 4 elements for better GPU utilization
    const int elementsPerThread = 4;
    for (int i = 0; i < elementsPerThread; i++)
    {
        int elementIdx = idx * elementsPerThread + i;
        if (elementIdx < size)
        {
            int value = input[elementIdx];
            int digit = (value >> bit) & 1;       // Extract current bit
            atomicAdd(&blockHistogram[digit], 1); // Atomic update of histogram
        }
    }
    __syncthreads();

    // Store block histogram to global memory
    if (threadIdx.x < 2)
    {
        int blockId = blockIdx.x;
        histogram[blockId * 2 + threadIdx.x] = blockHistogram[threadIdx.x];
    }
}

// Kernel to compute prefix sum (inclusive scan) for histogram
__global__ void computePrefixSum(int *input, int *output, int size)
{
    extern __shared__ int temp[]; // Dynamic shared memory allocation
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data from global to shared memory
    if (idx < size)
    {
        temp[tid] = input[idx];
    }
    else
    {
        temp[tid] = 0;
    }
    __syncthreads();

    // Parallel prefix sum algorithm
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid >= stride)
        {
            temp[tid] += temp[tid - stride];
        }
        __syncthreads();
    }

    // Write result back to global memory
    if (idx < size)
    {
        output[idx] = temp[tid];
    }
}

// Kernel to scatter elements to correct positions based on radix digit
__global__ void scatterElements(int *input, int *output, int *globalPrefix, int bit, int size, int numBlocks)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory for block offsets
    __shared__ int blockOffset[2];

    // Load block offsets from global prefix sum
    if (threadIdx.x < 2)
    {
        int blockId = blockIdx.x;
        // Get cumulative counts from previous blocks
        blockOffset[threadIdx.x] = (blockId > 0) ? globalPrefix[blockId * 2 + threadIdx.x - 1] : 0;
    }
    __syncthreads();

    // Each thread processes multiple elements
    const int elementsPerThread = 4;
    for (int i = 0; i < elementsPerThread; i++)
    {
        int elementIdx = idx * elementsPerThread + i;
        if (elementIdx < size)
        {
            int value = input[elementIdx];
            int digit = (value >> bit) & 1; // Extract current bit

            // Calculate correct position in output array
            int position;
            if (digit == 0)
            {
                // Place in zeros section
                position = blockOffset[0];
                blockOffset[0]++;
            }
            else
            {
                // Place in ones section (after all zeros)
                int totalZeros = globalPrefix[numBlocks * 2 - 1]; // Last element of prefix sum
                position = totalZeros + blockOffset[1];
                blockOffset[1]++;
            }

            // Scatter element to correct position
            if (position < size)
            {
                output[position] = value;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int *array;               // Pointer to input array
    int size, seed;           // Array size and random seed
    bool printSorted = false; // Flag to control output printing

    // Command line argument validation
    if (argc < 4)
    {
        std::cerr << "usage: " << argv[0]
                  << " [amount of random nums to generate] [seed value for rand]"
                  << " [1 to print sorted array, 0 otherwise]" << std::endl;
        exit(-1);
    }

    // Parse array size from command line
    {
        std::stringstream ss1(argv[1]);
        ss1 >> size;
    }
    // Parse random seed from command line
    {
        std::stringstream ss1(argv[2]);
        ss1 >> seed;
    }
    // Parse print flag from command line
    {
        int sortPrint;
        std::stringstream ss1(argv[3]);
        ss1 >> sortPrint;
        if (sortPrint == 1)
            printSorted = true;
    }

    // Generate random input array
    array = makeRandArray(size, seed);

    // CUDA events for timing measurement
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);

    // Device memory pointers
    int *d_array, *d_temp, *d_histogram, *d_prefix;

    // Allocate device memory
    CudaSafeCall(cudaMalloc(&d_array, size * sizeof(int)));
    CudaSafeCall(cudaMalloc(&d_temp, size * sizeof(int)));

    // Calculate GPU execution configuration
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Ensure massive parallelism for large arrays (meets project requirement)
    if (size >= 1000000)
    {
        numBlocks = 1024;      // Use 1024 blocks for maximum parallelism
        threadsPerBlock = 256; // Total threads: 1024 * 256 = 262,144
    }

    // Allocate memory for histogram and prefix sum arrays
    int histogramSize = numBlocks * 2; // 2 digits (0 and 1) per block
    CudaSafeCall(cudaMalloc(&d_histogram, histogramSize * sizeof(int)));
    CudaSafeCall(cudaMalloc(&d_prefix, histogramSize * sizeof(int)));

    // Copy input data from host to device
    CudaSafeCall(cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice));

    // Buffer pointers for ping-pong between iterations
    int *current = d_array;
    int *next = d_temp;

    // Radix sort: process each bit from LSB to MSB (32 bits for integers)
    for (int bit = 0; bit < 32; bit++)
    {
        // Step 1: Compute histogram of digits for current bit
        computeHistogram<<<numBlocks, threadsPerBlock>>>(current, d_histogram, bit, size, numBlocks);
        CudaCheckError();

        // Step 2: Compute prefix sum of histogram (exclusive scan)
        computePrefixSum<<<1, histogramSize, histogramSize * sizeof(int)>>>(d_histogram, d_prefix, histogramSize);
        CudaCheckError();

        // Step 3: Scatter elements to correct positions based on digit
        scatterElements<<<numBlocks, threadsPerBlock>>>(current, next, d_prefix, bit, size, numBlocks);
        CudaCheckError();

        // Swap buffers for next iteration
        int *temp = current;
        current = next;
        next = temp;
    }

    // Ensure final result is in d_array
    if (current != d_array)
    {
        CudaSafeCall(cudaMemcpy(d_array, current, size * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    // Copy sorted result back to host
    CudaSafeCall(cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CudaSafeCall(cudaFree(d_array));
    CudaSafeCall(cudaFree(d_temp));
    CudaSafeCall(cudaFree(d_histogram));
    CudaSafeCall(cudaFree(d_prefix));

    // Stop timer and calculate elapsed time
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);

    // Print execution time (required by project)
    std::cerr << "Total time in seconds: " << timeTotal / 1000.0 << std::endl;

    // Optional: Print sorted array if requested
    if (printSorted)
    {
        int elementsToShow = (size > 100) ? 100 : size;
        for (int i = 0; i < elementsToShow; i++)
        {
            std::cout << array[i] << " ";
        }
        if (size > 100)
        {
            std::cout << "... (showing first 100 of " << size << " elements)";
        }
        std::cout << std::endl;
    }

    // Cleanup host memory
    delete[] array;
    return 0;
}