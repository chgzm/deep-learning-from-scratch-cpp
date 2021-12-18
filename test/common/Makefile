CXX := g++
CXXFLAGS := -Wall -O3 -std=c++17
INCLUDE := -I../../common
LIB := -lgtest -lgtest_main -lpthread

SRCS := $(wildcard ../../common/*.cpp)
#SRCS += $(wildcard *.cpp)
SRCS += TestFunction.cpp
SRCS += TestMNIST.cpp
SRCS += TestMatrix.cpp
SRCS += TestByteReader.cpp
SRCS += TestRandom.cpp
SRCS += TestMmapper.cpp
OBJS := $(SRCS:%.cpp=%.o)

TARGET := gtest

$(TARGET): $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $(OBJS) $(LIB)

%.o: %.cpp
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c $< -o $@ $(LIBS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
