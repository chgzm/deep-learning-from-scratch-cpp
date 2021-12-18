CXX := g++
CXXFLAGS := -Wall -O3 -std=c++17 -fopenmp 
INCLUDE := -I./../common/

SRCS := $(wildcard ./../common/*.cpp)
SRCS += neuralnet_mnist_batch.cpp
OBJS := $(SRCS:.cpp=.o)
LIBS := -lm
TARGET := neuralnet_mnist_batch

$(TARGET): $(OBJS)
	$(CXX) $(INCLUDE) -o $@ $(OBJS) $(CXXFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)

