CXX        = g++
OCV_LIBS   = -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_highgui
CXXFLAGS   = -std=c++17 -O3 -march=native -Xpreprocessor -fopenmp \
             -I$(shell brew --prefix libomp)/include \
             -I$(shell brew --prefix opencv)/include/opencv4
LDFLAGS    = -L$(shell brew --prefix libomp)/lib -lomp \
             -L$(shell brew --prefix opencv)/lib $(OCV_LIBS)

all: time_mirror

time_mirror: time_mirror.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: time_mirror
	./time_mirror

dist: time_mirror
	rm -rf dist
	mkdir -p dist/libs
	cp time_mirror dist/
	dylibbundler -b -x dist/time_mirror -d dist/libs/ -p @executable_path/libs/
	zip -r9 time_mirror_mac.zip dist/
	@echo "Built: time_mirror_mac.zip ($$(du -sh time_mirror_mac.zip | cut -f1))"

clean:
	rm -f time_mirror
	rm -rf dist time_mirror_mac.zip

.PHONY: all run dist clean
