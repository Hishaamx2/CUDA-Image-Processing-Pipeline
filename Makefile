export PATH := /usr/local/cuda-13/bin:$(PATH)
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80
TARGET = gpu_image_pipeline
SRC = src/main.cu

all:
	$(NVCC) $(SRC) -o $(TARGET) $(NVCC_FLAGS)
clean:
	rm -f $(TARGET)
