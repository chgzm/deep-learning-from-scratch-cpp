CXX := g++
CXXFLAGS := -Wall -O3 -std=c++17 -fopenmp

SRCS := $(wildcard ./*.cpp)
OBJS := $(SRCS:.cpp=.o)

all: $(OBJS)

.PHONY: clean
clean:
	rm -f *.o
