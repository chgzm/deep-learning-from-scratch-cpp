CXX := g++
CXXFLAGS := -Wall -O3 -std=c++17 -fopenmp 
INCLUDE := -I./../common/

SRCS := $(wildcard ./../common/*.cpp)
SRCS += NeuralnetMNISTBatch.cpp
OBJS := $(SRCS:.cpp=.o)
LIBS := -lm
TARGET := NeuralnetMNISTBatch

$(TARGET): $(OBJS)
	$(CXX) $(INCLUDE) -o $@ $(OBJS) $(CXXFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)

