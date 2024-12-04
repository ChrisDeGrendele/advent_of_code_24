NVCC = nvcc
NVCC_FLAGS = -O3 --std=c++14

# Change these if your CUDA installation is in a different location
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include

# Target executable
TARGET = day1.ex

# Source files
SOURCES = day1.cu kernels.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INC_PATH) $^ -o $@

clean:
	rm -f $(TARGET)