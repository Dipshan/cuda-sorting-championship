# Project - Tour d’Algorithms: Cuda Sorting Championship

### Authors: 
    Deepshan Adhikari
    deepadh@siue.edu
    Student ID: 800846035

    Sabin Ghimire
    

    Sumit Shrestha

---

### Project Description:
This project implements three different sorting algorithms using CUDA:
1. thrust - Uses Thrust library for sorting (baseline)
2. singlethread - Single-threaded bubble sort on GPU
3. multithread - Parallel bitonic sort on GPU

---

### Project Structure:
```
cuda-sorting-championship/
├──sort
│    ├── multithread_sort.cu
│    ├── singlethread_sort.cu
│    ├── thrust_sort.cu
├── Makefile                          // Make build configuration
├── README.md                        // Project documentation (this file)
```

---

### Usage:
```
./build/thrust [num_elements] [seed] [print_flag]
./build/singlethread [num_elements] [seed] [print_flag]
./build/multithread [num_elements] [seed] [print_flag]
```

---

### Parameters:
- num_elements: Number of random integers to generate and sort
- seed: Seed value for random number generation
- print_flag: 1 to print sorted array, 0 otherwise

---

### Examples:
```
./build/thrust 1000000 42 0
./build/singlethread 10000 123 1
./build/multithread 1000000 456 0
```