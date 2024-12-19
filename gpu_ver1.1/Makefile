TARGET = ann
NVCC = nvcc
CXXFLAGS = -std=c++11 -Iheader

SRCS = main.cu src/ActivationFunction.cu src/ANN.cu src/DenseLayer.cu src/Kernel.cu src/loader.cu src/Macro.cu
OBJS = $(SRCS:.cu=.o)

all: $(TARGET) 

$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) -o $@ $^ && ./$(TARGET)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
