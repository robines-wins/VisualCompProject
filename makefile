OPENCV_CFLAGS = $(shell pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv)

CXX = g++ -std=c++11 -O2

default: main

QMULset.o: QMULset.cpp QMULset.h
	$(CXX) $(OPENCV_CFLAGS) -c QMULset.cpp

HPset.o: HPset.cpp HPset.h
	$(CXX) $(OPENCV_CFLAGS) -c HPset.cpp

EigenFaces.o: EigenFaces.cpp EigenFaces.h
	$(CXX) $(OPENCV_CFLAGS) -c EigenFaces.cpp

EigenFacePoseEstimation.o: EigenFaces.h EigenFacePoseEstimation.cpp EigenFacePoseEstimation.h
	$(CXX) $(OPENCV_CFLAGS) -c EigenFacePoseEstimation.cpp

answer.o: QMULset.h HPset.h EigenFaces.h EigenFacePoseEstimation.h answer.cpp answer.h
	$(CXX) $(OPENCV_CFLAGS) -c answer.cpp

main.o: QMULset.h HPset.h EigenFaces.h EigenFacePoseEstimation.h answer.h main.cpp main.h
	$(CXX) $(OPENCV_CFLAGS) -c main.cpp

main: QMULset.o HPset.o EigenFaces.o EigenFacePoseEstimation.o answer.o main.o
	$(CXX) $(OPENCV_CFLAGS) $(OPENCV_LIBS) -o main QMULset.o HPset.o EigenFaces.o EigenFacePoseEstimation.o answer.o main.o

run: main
	./main

clean:
	rm *.o
