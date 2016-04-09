OPENCV_CFLAGS = $(shell pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv)

CXX = g++ -std=c++11 -O2

default: main

QMULset.o: QMULset.cpp QMULset.h
	$(CXX) $(OPENCV_CFLAGS) -c QMULset.cpp

EigenFaces.o: EigenFaces.cpp EigenFaces.h
	$(CXX) $(OPENCV_CFLAGS) -c EigenFaces.cpp

main.o: main.cpp main.h
	$(CXX) $(OPENCV_CFLAGS) -c main.cpp

main: QMULset.o EigenFaces.o main.o
	$(CXX) $(OPENCV_CFLAGS) $(OPENCV_LIBS) -o main QMULset.o EigenFaces.o main.o

run: main
	./main

clean:
	rm *.o
