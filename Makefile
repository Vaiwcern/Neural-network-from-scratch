NVCC = nvcc

TARGET = ann_cuda

SRC = main1.cpp kernel.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
