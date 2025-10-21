# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O2 -Wall
NVCCFLAGS = -std=c++11 -O2

# Check if CUDA is available
CUDA_CHECK := $(shell which nvcc 2>/dev/null)
ifdef CUDA_CHECK
    COMPILER = $(NVCC)
    FLAGS = $(NVCCFLAGS)
    CUDA_AVAILABLE = 1
else
    COMPILER = $(CXX)
    FLAGS = $(CXXFLAGS) -DNO_CUDA
    CUDA_AVAILABLE = 0
endif

# Source files
THRUST_SRC = sort/thrust_sort.cu
SINGLE_SRC = sort/singlethread_sort.cu
MULTI_SRC = sort/multithread_sort.cu

# Executable names
THRUST_EXE = thrust
SINGLE_EXE = singlethread
MULTI_EXE = multithread

# Default target - all
all: $(THRUST_EXE) $(SINGLE_EXE) $(MULTI_EXE)

# Default target - only build thrust
thrust: $(THRUST_EXE)

# Thrust sort executable
$(THRUST_EXE): $(THRUST_SRC)
	@echo "CUDA available: $(CUDA_AVAILABLE)"
	$(COMPILER) $(FLAGS) -o $@ $<

# Single thread sort executable  
$(SINGLE_EXE): $(SINGLE_SRC)
	$(COMPILER) $(FLAGS) -o $@ $<

# Multi thread sort executable
$(MULTI_EXE): $(MULTI_SRC)
	$(COMPILER) $(FLAGS) -o $@ $<

# Clean build files
clean:
	rm -f $(THRUST_EXE) $(SINGLE_EXE) $(MULTI_EXE)

# Phony targets
.PHONY: all clean