NVCC = nvcc
TARGET = ann_cuda

SRC = main.cpp kernel.cu loader.cpp ann_cpu.cpp
HEADERS = loader.h ann_cpu.h

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
