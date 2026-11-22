# Makefile — DCF-Meow AVX2 build, tuned for i5-1335U
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): compiles the super-fast C++ math file
# into a shared library Python can call, using every speed trick this CPU has.
CXX      ?= g++
# -O3: max optimize. -march=native: use THIS cpu's best instructions (AVX2+FMA+BMI2 on Raptor Lake).
# -mtune=native: schedule for i5-1335U. -mfma: fused multiply-add (1 instr for a*b+c, DCF core).
# -flto: link-time optimize across files. -funroll-loops: unroll DCF year loop. -fopenmp: multi-thread grid/MC.
# -ffast-math: allow reordering (safe here, no NaN/Inf in DCF). -shared -fPIC: make Python-loadable .so.
CXXFLAGS := -O3 -march=native -mtune=native -mavx2 -mfma -mbmi -mbmi2 \
            -flto -funroll-loops -ffast-math -fopenmp -fPIC -std=c++17 -Wall -Wextra
LDFLAGS  := -shared -flto -fopenmp
SRC := cpp/dcf_avx2.cpp
OUT := build/libdcf_avx2.so

all: $(OUT)

$(OUT): $(SRC) cpp/dcf_avx2.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $(OUT) $(SRC) $(LDFLAGS)
	@echo "Built $@ — check AVX2: objdump -d $@ | grep -c vfmadd"

clean:
	rm -f $(OUT)
	rm -rf build/__pycache__

.PHONY: all clean
