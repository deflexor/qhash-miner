# CPU-only Makefile (use CMake when CUDA toolkit is available)
CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Iinclude -DQHASH_CPU_ONLY
LDFLAGS ?= -lm

BUILD := build-cpu
OBJS := $(BUILD)/qhash_cpu.o $(BUILD)/qhash_kernel.o

.PHONY: all test bench clean

all: $(BUILD)/qhash-miner $(BUILD)/qhash-test $(BUILD)/qhash-bench

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/qhash_cpu.o: src/qhash_cpu.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/qhash_kernel.o: src/qhash_kernel.cu | $(BUILD)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

$(BUILD)/qhash-miner: src/main.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD)/qhash-test: tests/test_circuit.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD)/qhash-bench: bench/benchmark.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

test: $(BUILD)/qhash-test $(BUILD)/qhash-miner
	$(BUILD)/qhash-test
	$(BUILD)/qhash-miner --self-test

bench: $(BUILD)/qhash-bench
	$(BUILD)/qhash-bench 16

clean:
	rm -rf $(BUILD)
