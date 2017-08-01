CXX= g++
CXXFLAGS= -std=c++11 -O3 -march=native -fopenmp -DNDEBUG -I$(CONDA_PREFIX)/include -Wno-depracated-declarations

LDFLAGS=-fopenmp $(BLAS_LIBS) -Wl,-rpath,$(CONDA_PREFIX)/lib

PROGS = prog-1 prog-2

all: $(PROGS)

clean:
	rm -f *.o $(PROGS)

.PHONY: all clean
