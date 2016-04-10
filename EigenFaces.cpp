#include <iostream>
#include <opencv2/opencv.hpp>
#include "EigenFaces.h"

using namespace cv;
using namespace std;
//WARNING: all of this function are creating/working with row vector

/*
static bool ANSWER_Q  = true;
static bool inri  = false;
static int ri = 0;*/

Mat imagesToPcaMatrix(vector<Mat> imgSet){
    Mat dataM = Mat(0, imgSet.front().rows*imgSet.front().cols, imgSet.front().type());
    for (auto it = imgSet.begin(); it != imgSet.end(); it++) {
        Mat vectorized = it->clone().reshape(0, 1);
        dataM.push_back(vectorized);
    }
    dataM.convertTo(dataM, CV_64F);
    return dataM;
}

Mat project(const Mat& vector, const Mat& base){
    Mat projection = Mat(1, base.rows, base.type());
    assert(vector.rows == 1);
    
    for (int i =0; i<base.rows; i++) { //compute each coordinate according to the course formula
        projection.at<double>(0,i) = vector.dot(base.row(i));
    }
    return projection;
}

Mat backproject(const Mat& vector, const Mat& base){
    assert(vector.rows == 1);
    return vector*base; //just a base transition because here we have the transition matrix
}

Mat computeEigenBase(Mat& data, int numOfComp){
    Mat DtD, EValues, EVector;
    mulTransposed(data, DtD, true); //eigen work with collum vector so we temporary transpose our matrix of row vectors
    eigen(DtD, EValues, EVector);
    
    //if (inri) {outputEVectors(EVector);}
    
    Mat base;
    transpose(EVector.colRange(0, numOfComp), base); //transpose matrix eigenvector to get back to row vector and selecting only the number we want
    base.convertTo(base, data.type()); //make sure type of base and vector are the same
    
    return base;
}

double reconstructionError(Mat& traningM, Mat& testingM, int numOfComp){
    
    //calculation of mean vector, and substract it from each row vector
    Mat mean = Mat(1, traningM.cols, traningM.type());
    reduce(traningM, mean, 0, CV_REDUCE_AVG);
    
    for (int i=0; i<traningM.rows; i++) {
        traningM.row(i) -= mean;
        
    }
    
    
    Mat base = computeEigenBase(traningM, numOfComp);
    
    double reconstructionError = 0.0;
    
    for (int i = 0; i<testingM.rows; i++) {
        Mat temp,result;
        //project
        temp = project(testingM.row(i), base);
        //back
        result = backproject(temp, base);
        
        reconstructionError += norm(testingM.row(i),result,NORM_L2);
    }
    reconstructionError = reconstructionError/testingM.rows;
    return reconstructionError;
    
}

double kFoldCrossValidation(vector<Mat> imgSet, int numOfComp, int k,bool testWithtest){
    
    
    random_shuffle(imgSet.begin(), imgSet.end()); //shuffle the order of images
    double reconstructionerror = 0.0; //initiate the error
    
    Mat dataM = imagesToPcaMatrix(imgSet); //transform vector of image to matrix of row vector
    assert(dataM.cols == imgSet.front().rows*imgSet.front().cols);
    assert(dataM.rows == imgSet.size());
    
    for (int i=0; i<k; i++) { //for each fold
        Mat test,train;
        splitM(dataM, test, train, k, i);
        //compute the reconstruction error for this configuration
        reconstructionerror += testWithtest ? reconstructionError(train, test, numOfComp) : reconstructionError(train, train, numOfComp);
    }
    
    //transform sum of error to an average
    reconstructionerror = reconstructionerror/(double)k;
    
    return reconstructionerror;
    
    
}

void splitM(Mat& dataM, Mat& test, Mat& train,int k, int i){
    int imgPerFolds = (int)dataM.rows/k;
    Mat trainA, trainB;
    
    dataM.rowRange(i*imgPerFolds, (i+1)*imgPerFolds).copyTo(test); //create the matrix of test vector
    assert(test.rows == imgPerFolds);
    
    //create matrxix of training vector specila case on extraction of vectores with 1st and last fold
    if (i == 0) {
        dataM.rowRange((i+1)*imgPerFolds, k*imgPerFolds).copyTo(train);
    }
    else if (i == k-1){
        dataM.rowRange(0, i*imgPerFolds).copyTo(train);
    }
    else{
        dataM.rowRange(0, i*imgPerFolds).copyTo(trainA);
        dataM.rowRange((i+1)*imgPerFolds, k*imgPerFolds).copyTo(trainB);
        dataM(Range(0, i*imgPerFolds),Range(0, dataM.cols)).copyTo(trainA);
        dataM(Range((i+1)*imgPerFolds, k*imgPerFolds),Range(0, dataM.cols)).copyTo(trainB);
        vconcat(trainA, trainB, train);
    }
    
    assert(train.cols == dataM.cols);
    assert(train.rows == (k-1)*imgPerFolds);
    assert(test.rows == imgPerFolds);
    
}

void answerQ3(vector<Mat> set){
    
    int numC[] = {1,2,5,10,20,50,100,200,500,1000,2000,3000,set.front().rows*set.front().cols};
    for (int i = 0; i<13; i++) {
        cout<< numC[i] <<endl;
    }
    cout <<endl;
    for (int i = 0; i< 13; i++) {
        cout << "k validation for reconstruction of training images" <<endl << endl;
        cout << kFoldCrossValidation(set, numC[i],7,false)<<endl;
    }
    cout <<endl;
    for (int i = 0; i< 13; i++) {
        cout << "k validation for reconstruction of testing images" <<endl << endl;
        cout << numC[i] <<" " << kFoldCrossValidation(set, numC[i]) <<endl;
    }
}

void answerQ4(vector<Mat> set){
    random_shuffle(set.begin(), set.end());
    Mat dataM = imagesToPcaMatrix(set);
    Mat test,train;
    splitM(dataM, test, train, 7, rand()%7);
    
    Mat mean = Mat(1, train.cols, train.type());
    reduce(train, mean, 0, CV_REDUCE_AVG);
    mean.reshape(1, 100).convertTo(mean, CV_8U);
    imwrite("/Users/Mac-Robin/Documents/CompVis/Project/Project/outputimages/means.bmp", mean);
    
    Mat base = computeEigenBase(train, 10);
    for (int i = 0; i<10; i++) {
        Mat toOutput;
        base.row(i).reshape(1, 100).convertTo(toOutput, CV_8U);
        imwrite("/Users/Mac-Robin/Documents/CompVis/Project/Project/outputimages/Evector" +to_string(i)+".bmp", toOutput);
    }
    
}

void answerQ5(vector<Mat> set, int optimalfromQ3){
    Mat dataM = imagesToPcaMatrix(set);
    Mat test,train;
    splitM(dataM, test, train, 7, rand()%7);
    
    int numOfComp[] = {1,5,optimalfromQ3};
    Mat randImg[4] = {train.row(rand()%train.rows), train.row(rand()%train.rows), test.row(rand()%test.rows), test.row(rand()%test.rows)};
    
    for (int i=0; i<3; i++) {
        Mat base = computeEigenBase(train, numOfComp[i]);
        for (int j=0; j<4; j++) {
            Mat toOutput = randImg[j].reshape(1, 100);
            toOutput.convertTo(toOutput, CV_8U);
            imwrite("/Users/Mac-Robin/Documents/CompVis/Project/Project/outputimages/random" +to_string(j)+ "_Original_" +to_string(numOfComp[i])+"vectors.bmp", toOutput);
            
            toOutput = backproject(project(randImg[j], base), base).reshape(1, 100);
            toOutput.convertTo(toOutput, CV_8U);
            imwrite("/Users/Mac-Robin/Documents/CompVis/Project/Project/outputimages/random" +to_string(j)+ "_reconstruct_" +to_string(numOfComp[i])+"vectors.bmp", toOutput);
        }
    }
    
    
}