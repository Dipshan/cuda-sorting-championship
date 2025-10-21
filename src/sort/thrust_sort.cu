#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

using namespace std;

/**********************************************************
 * **********************************************************
 * error checking stufff
 ***********************************************************
 ***********************************************************/
// Enable this for error checking
#define CUDA_CHECK_ERROR
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err,
                           const char *file, const int line)
{
#ifdef CUDA_CHECK_ERROR
#pragma warning(push)
#pragma warning(disable : 4127)
    do
    {
        if (cudaSuccess != err)
        {
            fprintf(stderr,
                    "cudaSafeCall() failed at %s:%i : %s\n",
                    file, line, cudaGetErrorString(err));
            exit(-1);
        }
    } while (0);
#pragma warning(pop)
#endif
    return;
}
inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_CHECK_ERROR
#pragma warning(push)
#pragma warning(disable : 4127)
    do
    {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr,
                    "cudaCheckError() failed at %s:%i : %s.\n",
                    file, line, cudaGetErrorString(err));
            exit(-1);
        }
        // More careful checking. However, this will affect performance.
        // Comment if not needed.
        err = cudaDeviceSynchronize();
        if (cudaSuccess != err)
        {
            fprintf(stderr,
                    "cudaCheckError() with sync failed at %s:%i : %s.\n",
                    file, line, cudaGetErrorString(err));
            exit(-1);
        }
    } while (0);
#pragma warning(pop)
#endif
    return;
}
/***************************************************************
 * **************************************************************
 * end of error checking stuff
 ****************************************************************
 ***************************************************************/

// function takes an array pointer, and the number of rows and cols in the array, and
// allocates and intializes the array to a bunch of random numbers
// Note that this function creates a 1D array that is a flattened 2D array
// to access data item data[i][j], you must can use data[(i*rows) + j]
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

//*******************************//
// your kernel here!!!!!!!!!!!!!!!!!
//*******************************//
__global__ void matavgKernel(...)
{
}

void printArray(const thrust::host_vector<int> &array, int size)
{
    for (int i = 0; i < size; ++i)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    int *h_array;   // the pointer to the array of rands on host
    int size, seed; // values for the size of the array
    bool printSorted = false;
    // and the seed for generating
    // random numbers
    // check the command line args
    if (argc < 3)
    {
        std::cerr << "usage: "
                  << argv[0]
                  << " [number of random integers to generate] [seed value for random number generation]"
                  << std::endl;
        exit(-1);
    }
    {
        // convert cstrings to ints
        std::stringstream ss1(argv[1]);
        ss1 >> size;
    }
    {
        std::stringstream ss1(argv[2]);
        ss1 >> seed;
    }
    {
    }

    // get the random numbers
    h_array = makeRandArray(size, seed);

    /***********************************
     * create a cuda timer to time execution
     **********************************/
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    CudaSafeCall(cudaEventCreate(&startTotal));
    CudaSafeCall(cudaEventCreate(&stopTotal));
    CudaSafeCall(cudaEventRecord(startTotal, 0));
    /***********************************
     * end of cuda timer creation
     **********************************/

    /////////////////////////////////////////////////////////////////////
    /////////////////////// YOUR CODE HERE ///////////////////////
    /////////////////////////////////////////////////////////////////////

    thrust::device_vector<int> d_array(h_array, h_array + size);

    thrust::sort(d_array.begin(), d_array.end());

    CudaCheckError();

    thrust::host_vector<int> h_result;
    if (printSorted)
    {
        h_result = d_array;
    }

    /////////////////////////////////////////////////////////////////////
    /////////////////////// END OF YOUR CODE ///////////////////////
    /////////////////////////////////////////////////////////////////////

    /***********************************
     * Stop and destroy the cuda timer
     **********************************/
    CudaSafeCall(cudaEventRecord(stopTotal, 0));
    CudaSafeCall(cudaEventSynchronize(stopTotal));
    CudaSafeCall(cudaEventElapsedTime(&timeTotal, startTotal, stopTotal));
    CudaSafeCall(cudaEventDestroy(startTotal));
    CudaSafeCall(cudaEventDestroy(stopTotal));
    /***********************************
     * end of cuda timer destruction
     **********************************/

    delete[] h_array;

    std::cerr << "Total time in seconds: "
              << timeTotal / 1000.0 << std::endl;

    if (printSorted)
    {
        ///////////////////////////////////////////////
        /// Your code to print the sorted array here //
        ///////////////////////////////////////////////
        printArray(h_result, size);
    }

    return 0;
}