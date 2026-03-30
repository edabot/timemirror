CXX      = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Xpreprocessor -fopenmp \
           -I$(shell brew --prefix libomp)/include
LDFLAGS  = -L$(shell brew --prefix libomp)/lib -lomp \
           $(shell pkg-config --cflags --libs opencv4)

all: time_mirror

time_mirror: time_mirror.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: time_mirror
	./time_mirror

clean:
	rm -f time_mirror

.PHONY: all run clean
