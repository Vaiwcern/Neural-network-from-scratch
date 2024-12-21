NVCC = nvcc
TARGET = ann_cuda

SRC = main.cu loader.cpp ann_cpu.cpp
HEADERS = loader.h ann_cpu.h

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) -o $(TARGET) $(SRC) && ./$(TARGET)

clean:
	rm -f $(TARGET)
