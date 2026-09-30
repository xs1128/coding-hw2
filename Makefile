# Build on the nano4 login node (no GPU needed to compile):
#   source scripts/env.sh && make -j
NVHPC_ROOT ?= /work/envstack/apps/nvhpc/26.3/Linux_x86_64/26.3
CUDA_PATH  ?= $(NVHPC_ROOT)/cuda/13.1
MATH_PATH  ?= $(NVHPC_ROOT)/math_libs/13.1
GPU_ARCH   ?= 90

NVCC    := $(CUDA_PATH)/bin/nvcc
NVFLAGS := -std=c++17 -O3 -lineinfo -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH) \
           -Iinclude -I$(MATH_PATH)/include -Xcompiler -fPIC
LDLIBS  := -L$(MATH_PATH)/lib64 -lcublas -lcublasLt -Xlinker -rpath=$(MATH_PATH)/lib64 \
           -Xlinker -rpath=$(CUDA_PATH)/lib64

BUILD := build
LIB   := $(BUILD)/libozdgemm.a

all: $(LIB) $(BUILD)/test_ozdgemm $(BUILD)/bench

$(BUILD):
	mkdir -p $@

$(BUILD)/ozdgemm.o: src/ozdgemm.cu include/ozdgemm.h | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(LIB): $(BUILD)/ozdgemm.o
	ar rcs $@ $^

$(BUILD)/test_ozdgemm: tests/test_ozdgemm.cu tests/common.cuh $(LIB)
	$(NVCC) $(NVFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD)/bench: bench/bench.cu tests/common.cuh $(LIB)
	$(NVCC) $(NVFLAGS) $< $(LIB) $(LDLIBS) -o $@

# DGEMM hook for netlib HPL (see scripts/build_hpl.sh)
$(BUILD)/libozdhpl.a: hpl/ozd_shim.cu include/ozdgemm.h | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $(BUILD)/ozd_shim.o
	ar rcs $@ $(BUILD)/ozd_shim.o

hpl-shim: $(LIB) $(BUILD)/libozdhpl.a

test: $(BUILD)/test_ozdgemm
	$(BUILD)/test_ozdgemm

clean:
	rm -rf $(BUILD)

.PHONY: all test clean hpl-shim
