CXX := g++
CXXFLAGS := -Wall -O3 -std=c++17 -fopenmp
INCLUDE := -I./../common/
LIBS := -lm

SRCS := $(wildcard ./../common/*.cpp)
OBJS := $(SRCS:.cpp=.o)

TARGETS := OptimizerCompareNaive 
TARGETS += OptimizerCompareMnist 
TARGETS += WeightInitActivationHistogram 
TARGETS += WeightInitCompare 
TARGETS += BatchNormTest 
TARGETS += OverfitWeightDecay
TARGETS += OverfitDropout 
TARGETS += HyperparameterOptimization

all: $(TARGETS) 

OptimizerCompareNaive: OptimizerCompareNaive.cpp $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $< $(OBJS) $(LIBS)

OptimizerCompareMnist: OptimizerCompareMnist.cpp $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $< $(OBJS) $(LIBS)

WeightInitActivationHistogram: WeightInitActivationHistogram.cpp $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $< $(OBJS) $(LIBS)

WeightInitCompare: WeightInitCompare.cpp $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $< $(OBJS) $(LIBS)

BatchNormTest: BatchNormTest.cpp $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $< $(OBJS) $(LIBS)

OverfitWeightDecay: OverfitWeightDecay.cpp $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $< $(OBJS) $(LIBS)

OverfitDropout: OverfitDropout.cpp $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $< $(OBJS) $(LIBS)

HyperparameterOptimization: HyperparameterOptimization.cpp $(OBJS)
	$(CXX) $(INCLUDE) $(CXXFLAGS) -o $@ $< $(OBJS) $(LIBS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJS)

